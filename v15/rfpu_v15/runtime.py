"""Batched, captured stock HF neural ALU. Explicit host instruction scheduling.

Torch operations outside layers encode/decode bits, pad, gather and copy data.
No external addition, multiplication, shifts that narrow numerical values, FFT,
convolution or inference computes DSP intermediates. Bit extraction and assembly
are representation conversions, analogous to v1.4 output-code projection.
"""
import os,time
from .neural import build,snapshot
from .paths import DATA
CHECKPOINT=os.environ.get('RFPU_V15_CHECKPOINT',str(DATA/'checkpoints/rfpu-v15/bitwise-I16-W64-v1'))

class Engine:
    def __init__(self,batch=128,path=CHECKPOINT):
        import torch
        from transformers import DynamicCache
        from rfpu.memory import gpu_guard,windows_gpu_memory
        from rfpu_v12.checkpoint import Processor
        from chipformer.rv.hf import head_slots
        self.torch=torch;self.batch=batch;self.c=build();snapshot(self.c)
        self.p=Processor(self.c,path);self.forwards=0;self.logical_elements=0;self.operations={}
        # Static per-layer DeltaNet states dominate batch memory; admit before allocation.
        gpu_guard(required_bytes=batch*self.c.n_layers()*16*128*128*4+1024**3)
        self.p.cache=DynamicCache(config=self.p.cfg);self.ids=torch.full((batch,1),self.p.manifest['zero_prefix'],device='cuda',dtype=torch.long)
        def forward():
            with torch.no_grad():
                h=self.p.embedding(self.ids)
                for layer in self.p.layers:h=layer(h,position_embeddings=None,attention_mask=None,position_ids=None,past_key_values=self.p.cache)
                return h[:,0,self.p.columns].float()
        start=time.perf_counter()
        forward();torch.cuda.synchronize()
        slots=head_slots(self.c);self.states=[]
        for n in self.c.arg_heads:
            index=next(i for i,node in enumerate(self.c.nodes) if node is n);layer,head=slots[index]
            self.states.append(self.p.cache.layers[layer].recurrent_states[0][:,head,0,:])
        self.bits=torch.arange(64,device='cuda');self.output_indices=[]
        for expr in self.c.output_bits:
            assert len(expr.terms)==1
            dim,coef=next(iter(expr.terms.items()));assert coef==1
            self.output_indices.append(dim-1)
        from chipformer.q35.ir import A_REF
        for state in self.states:state[:,0:2].fill_(A_REF)
        warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):forward()
        torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.static=forward()
        torch.cuda.synchronize();self.capture_seconds=time.perf_counter()-start
        self.index=torch.tensor(self.output_indices,device='cuda');self.baseline=windows_gpu_memory()
        self.cache_bytes=sum(t.numel()*t.element_size() for layer in self.p.cache.layers for name in ('conv_states','recurrent_states') for t in (getattr(layer,name).values() if isinstance(getattr(layer,name),dict) else getattr(layer,name)) if t is not None)

    def tensor(self,value):return self.torch.as_tensor(value,device='cuda',dtype=self.torch.int64)

    def op(self,opcode,a,b=None):
        import torch
        from chipformer.q35.ir import A_BIT
        with torch.no_grad():
            a=torch.as_tensor(a,device='cuda',dtype=torch.int64);b=torch.zeros_like(a) if b is None else torch.as_tensor(b,device='cuda',dtype=torch.int64)
            a,b=torch.broadcast_tensors(a,b);shape=a.shape;a=a.reshape(-1);b=b.reshape(-1);outputs=[]
            for start in range(0,a.numel(),self.batch):
                count=min(self.batch,a.numel()-start)
                for state,values in zip(self.states,(a,b)):
                    block=values[start:start+count]
                    if count<self.batch:block=torch.nn.functional.pad(block,(0,self.batch-count))
                    state[:,2:66].copy_(((block[:,None]>>self.bits)&1).float()*A_BIT)
                self.ids.fill_(opcode);self.graph.replay()
                bits=self.static[:,self.index].round().to(torch.int64)
                # Representation projection only: all arithmetic result bits came from layers.
                outputs.append(torch.sum(bits<<self.bits,dim=1)[:count])
                self.forwards+=1;self.logical_elements+=count
            self.operations[str(opcode)]=self.operations.get(str(opcode),0)+1
            return torch.cat(outputs).reshape(shape)

    def telemetry(self):
        import torch
        from rfpu.memory import gpu_guard,windows_gpu_memory,assert_no_shared_growth
        torch.cuda.synchronize();memory=gpu_guard();wddm=assert_no_shared_growth(self.baseline)
        return {'neural_forwards':self.forwards,'logical_ALU_elements':self.logical_elements,'operations':dict(self.operations),'cache_bytes':self.cache_bytes,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'memory':memory,'wddm':wddm,'checkpoint_sha256':self.p.manifest['sha256'],'batch':self.batch}
