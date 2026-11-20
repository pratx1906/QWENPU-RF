"""Analytical bit-sliced ALU. Derived from QwenPU MIT primitives, not CPU DSP.

Runtime operands are raw bits in two neural recurrent heads. Host selects an
instruction and moves arrays; all mathematical primitives execute in emitted layers.
"""
from chipformer.q35.prims import Circ
from chipformer.q35.ir import Target,A_BIT,A_REF,HD,REF_COLS
from chipformer.rv.core import add_nibbles
import numpy as np
ADD,SUB,MUL,ROUND,CMP=range(5)

def build():
    c=Circ(Target(key_heads=16,value_heads=16,intermediate=3584));f=c.token_feature
    # Token 0..4 selects arithmetic; tokens 32+s select rounded right shift s.
    ops=[f('decode.op'+str(i),lambda t,i=i:int(t==i)) for i in range(5)]
    shifts=[f('decode.shift'+str(i),lambda t,i=i:int(t==32+i)) for i in range(31)]
    args=[]
    for name in ('A','B'):
        bits=c.slot_head('operand.'+name,{}, {0:c.one}, {2+i:0*c.one for i in range(64)},0*c.one)
        args.append(bits)
    c.arg_heads=[n for n in c.nodes if getattr(n,'name','').startswith('operand.')]
    a,b=args
    add=add_nibbles(c,a,b,16,[],'add64')
    negative_b=add_nibbles(c,[1-x for x in b],[1]+[0]*63,16,[],'negate_B')
    sub=add_nibbles(c,a,negative_b,16,[],'subtract64')
    # Baugh-Wooley two's-complement 16x16 product. All materialized values are bits.
    rows=[]
    for j in range(16):
        row=[0*c.one]*32
        for i in range(16):
            p=c.and_([a[i],b[j]],f'product.pp{i}_{j}')
            row[i+j]=1-p if (i==15)!=(j==15) else p
        rows.append(row)
    constant=[0*c.one]*32;constant[16]=c.one;constant[31]=c.one;rows.append(constant)
    level=0
    while len(rows)>1:
        rows=[add_nibbles(c,rows[i],rows[i+1],8,[],f'product.tree{level}_{i}') if i+1<len(rows) else rows[i] for i in range(0,len(rows),2)];level+=1
    product=rows[0]+[rows[0][-1]]*32
    # Round nearest, ties toward +infinity: (A+half)>>s. Defined modulo64 before shift.
    half=[sum(shifts[s] for s in range(1,31) if s-1==i) for i in range(64)]
    biased=add_nibbles(c,a,half,16,[],'round.bias')
    shifted=c.select_bits([([shifts[s]],biased[s:]+[biased[-1]]*s) for s in range(31)],64,'round.shift')
    different_sign=c.and_([a[-1],1-b[-1]],'compare.different_negative')
    same_sign=c.node(c.t_eq(a[-1]-b[-1],0,-1,1),'compare.same_sign')
    less=c.node(c.t_and([same_sign,sub[-1]])+c.t_const(0)+[(1,different_sign,c.one)],'compare.less')
    ge=[1-less]+[0*c.one]*63
    c.output_bits=c.select_bits([([ops[ADD]],add),([ops[SUB]],sub),([ops[MUL]],product),([sum(shifts)],shifted),([ops[CMP]],ge)],64,'result')
    c.next_bits=c.output_bits;c.max_token=62
    return c

def snapshot(c,a=0,b=0):
    for node,value in zip(c.arg_heads,(a,b)):
        row=np.zeros(HD);row[list(REF_COLS)]=A_REF
        for i in range(64):row[2+i]=A_BIT*((int(value)>>i)&1)
        node.init=[{0:row}]
