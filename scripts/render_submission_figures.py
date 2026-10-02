"""Render submission figures from saved results; never train or recompute scores.

Run from the repository root. Dependencies: matplotlib, numpy, pypdf, PyMuPDF.
Maps/scatter plots retain the original PDF vectors and point coordinates.
"""
from pathlib import Path
import csv
import json
import io
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.figure-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, TextStringObject, ArrayObject

OUT = ROOT / 'paper/figures'
WIDTH = 119 / 25.4
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9,
                     'axes.labelsize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8.5, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})

METHODS = [('emos_gl','EMOS'),('emos_bst_gl','Boosted EMOS'),
           ('emos_nn1','EMOS-NN1'),('emos_idw','EMOS-IDW'),
           ('emos_reg','EMOS-Reg'),('emos_regidw','EMOS-RegIDW'),
           ('samos_lin','SAMOS-Lin'),('nn_unk','NN-UNK'),('nn_knn','NN-kNN'),
           ('nn_noemb','NN-NoEmb'),('nn_attr','NN-Attr'),('nn_hybrid','NN-Hybrid'),
           ('nn_hybrid_knn','NN-Hybrid-kNN'),('nn_hybrid_res','NN-Hybrid-Res'),
           ('drn_lak','DRN'),('gnn_geo','GNN-Geo'),
           ('samos_mlp','SAMOS-MLP'),('samos_attn','SAMOS-Attn')]

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(OUT / name, metadata={'Creator':'Saved-results layout renderer',
                                     'Subject':'Layout revision; no new experiment'})
    plt.close(fig)

def folds():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH,5.2))
    colors = ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00','#444444']
    for row,(ds,title) in enumerate([('german','German (535 stations)'),('euppbench','EUPPBench (117 stations)')]):
        for col,design in enumerate(['random','spatial']):
            ax=axes[row,col]
            rows=read_csv(ROOT/f'figure_data/{ds}_{design}_folds.csv')
            for fold in range(7):
                sub=[r for r in rows if int(r['fold'])==fold]
                ax.scatter([float(r['lon']) for r in sub],[float(r['lat']) for r in sub],
                           s=5 if ds=='german' else 8,c=colors[fold],label=str(fold+1),linewidths=0)
            ax.set_title(('Random' if col==0 else 'Spatial')+'\n'+title,pad=7)
            ax.set_xlabel('Longitude (°E)')
            if col==0: ax.set_ylabel('Latitude (°N)')
            ax.set_aspect(1.35)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=7,title='Fold',frameon=False,
               handletextpad=.25,columnspacing=.65,markerscale=1.8)
    fig.subplots_adjust(left=.13,right=.99,bottom=.15,top=.90,hspace=.64,wspace=.45)
    save(fig,'Fig1.pdf')

def spatial_scores():
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,4.6),sharey=True)
    for ax,ds,title in zip(axes,['german','euppbench'],['German','EUPPBench']):
        data={r['method']:r for r in read_csv(ROOT/f'results/stage5/eval/final/summary_{ds}_spatial.csv')}
        for i,(key,label) in enumerate(METHODS):
            color='#0072B2' if int(data[key]['n_seeds'])==10 else '#555555'
            ax.plot(float(data[key]['crps']),i,'o',color=color,ms=4)
        ax.axvline(float(data['emos_bst_gl']['crps']),color='#777777',ls=':',lw=1)
        ax.set_title(title)
        ax.set_xlabel('CRPS (°C)')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis='x',alpha=.18)
        ax.set_ylim(len(METHODS)-.4,-.6)
        ax.set_xlim(.855,1.18)
    axes[0].set_yticks(range(len(METHODS)),[v[1] for v in METHODS])
    axes[1].tick_params(axis='y',length=0)
    fig.subplots_adjust(left=.32,right=.985,bottom=.13,top=.93,wspace=.48)
    save(fig,'Fig2.pdf')

def did():
    """Display saved matched/forecast-only contrasts before the original primary analysis."""
    reviews = {ds: json.loads((ROOT / f'results/review_supplementary/did_{ds}.json').read_text())
               for ds in ['euppbench', 'german']}
    matched = [('euppbench', 'matched_3v3_drn', 'EUPPBench: DRN'),
               ('euppbench', 'matched_3v3_gnn', 'EUPPBench: GNN-Geo'),
               ('euppbench', 'forecast_only_3v3_gnn', 'EUPPBench: GNN-Geo\n(forecast-only)'),
               ('german', 'matched_3v3_drn', 'German: DRN'),
               ('german', 'matched_3v3_gnn', 'German: GNN-Geo')]
    primary = [('euppbench', 'primary_10v3_drn', 'EUPPBench: DRN'),
               ('euppbench', 'primary_10v3_gnn', 'EUPPBench: GNN-Geo'),
               ('german', 'primary_10v3_drn', 'German: DRN'),
               ('german', 'primary_10v3_gnn', 'German: GNN-Geo')]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 4.75), sharex=True,
                             gridspec_kw={'height_ratios': [1.25, 1]})
    panels = [(axes[0], matched, (3, 3),
               '(a) Matched ensembles (post hoc)\n3 spatial / 3 random seeds'),
              (axes[1], primary, (10, 3),
               '(b) Original primary analysis\n10 spatial / 3 random seeds')]
    plotted = []
    for ax, entries, seeds, title in panels:
        for i, (ds, contrast, label) in enumerate(entries):
            d = reviews[ds]['contrasts'][contrast]['station']
            assert (d['n_seeds_spatial'], d['n_seeds_random']) == seeds
            x, lo, hi = d['did'], d['lo'], d['hi']
            assert lo <= x <= hi
            forecast = contrast == 'forecast_only_3v3_gnn'
            color = '#D55E00' if forecast else '#0072B2' if ds == 'euppbench' else '#555555'
            ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt='s' if forecast else 'o',
                        capsize=3, color=color, mfc='white' if forecast else color, ms=4)
            plotted.append({'dataset': ds, 'contrast': contrast,
                            'did': x, 'lo': lo, 'hi': hi,
                            'n_seeds_spatial': seeds[0], 'n_seeds_random': seeds[1]})
        ax.axvline(0, color='#777777', ls=':', lw=1)
        ax.set_yticks(range(len(entries)), [e[2] for e in entries])
        ax.set_ylim(len(entries)-.55, -.55)
        ax.set_xlim(-.025, .36)
        ax.set_title(title, loc='left', pad=9, fontsize=8.5)
        ax.grid(axis='x', alpha=.18)
    axes[1].set_xticks([0, .1, .2, .3])
    axes[1].set_xlabel('Spatial - random DiD (°C)\n(relative to Boosted EMOS)')
    fig.subplots_adjust(left=.43, right=.985, bottom=.16, top=.89, hspace=.60)
    save(fig, 'Fig3.pdf')
    return plotted

def original_three_seed_scores():
    data=json.loads((OUT/'derived_numbers.json').read_text())['exact_scores']
    methods=[m for m in METHODS if m[0] in data['german:random']]
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.9),sharey=True)
    for ax,ds,title in zip(axes,['german','euppbench'],['German','EUPPBench']):
        for i,(key,label) in enumerate(methods):
            for design,dy,marker,color in [('random',-.12,'o','#0072B2'),('spatial',.12,'s','#D55E00')]:
                ax.plot(data[f'{ds}:{design}'][key]['crps'],i+dy,marker,color=color,ms=3.7,
                        markerfacecolor='white' if design=='random' else color,
                        label=design.capitalize() if i==0 else None)
        ax.set_title(title)
        ax.set_xlabel('CRPS (°C)')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis='x',alpha=.18)
        ax.set_ylim(len(methods)-.4,-.6)
        ax.set_xlim(.78,1.185)
    axes[0].set_yticks(range(len(methods)),[v[1] for v in methods])
    axes[1].tick_params(axis='y',length=0)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(left=.32,right=.985,bottom=.21,top=.93,wspace=.48)
    save(fig,'FigS1.pdf')

def edit_vector_panels(source, clips, start, numbered=False):
    original=pymupdf.open(source)
    spans=[s for b in original[0].get_text('dict')['blocks'] for line in b.get('lines',[])
           for s in line['spans'] if s['size']<=6.1]
    lines=[line for b in original[0].get_text('dict')['blocks'] for line in b.get('lines',[])]
    rotated=[line for line in lines if line['dir']==(0.,-1.)]
    legend=[line for line in lines if any('training range' in s['text'] or 'outside range' in s['text'] for s in line['spans'])]
    horizontal=[line for line in lines if any('Station altitude' in s['text'] for s in line['spans'])]
    # Re-typeset complete rotated labels, rather than stretching glyph positioning
    # in the original vector text. Plot paths and markers are not redacted.
    for line in rotated+legend:
        original[0].add_redact_annot(pymupdf.Rect(line['bbox']),fill=None)
    # Font bounding boxes of the original xlabel overlap neighbouring tick
    # labels. Use a narrow strip at its baseline to retain every tick label.
    for line in horizontal:
        x0,_,x1,_=line['bbox']; baseline=line['spans'][0]['origin'][1]
        original[0].add_redact_annot(pymupdf.Rect(x0,baseline-6,x1,baseline+1),fill=None)
    original[0].apply_redactions(images=0,graphics=0,text=0)
    reader=PdfReader(io.BytesIO(original.tobytes()))
    page=reader.pages[0]
    contents=ContentStream(page.get_contents(),reader)
    size=None
    for args,op in contents.operations:
        if op==b'Tf':
            size=float(args[1])
            if size>6.1 and size<9: args[1]=FloatObject(9)
        elif size is not None and size<=6.1 and op in (b'Tj',b'TJ'):
            # Remove only small annotation text, retaining plot marks and other text.
            args[0]=TextStringObject('') if op==b'Tj' else ArrayObject()
    page.replace_contents(contents)
    writer=PdfWriter(); writer.add_page(page)
    buf=io.BytesIO(); writer.write(buf)
    edited=pymupdf.open(stream=buf.getvalue(),filetype='pdf')
    for i,coords in enumerate(clips):
        clip=pymupdf.Rect(coords)
        panel=pymupdf.open(); p=panel.new_page(width=clip.width,height=clip.height)
        p.show_pdf_page(p.rect,edited,0,clip=clip)
        for line in rotated:
            point=pymupdf.Point(line['spans'][0]['origin'])
            if not clip.contains(point): continue
            text=''.join(s['text'] for s in line['spans']).replace('−','-').replace('†','')
            x=point.x-clip.x0; center=(line['bbox'][1]+line['bbox'][3])/2-clip.y0
            length=pymupdf.get_text_length(text,fontname='helv',fontsize=9)
            p.insert_text((x,center+length/2),text,fontname='helv',fontsize=9,rotate=90)
        for line in legend:
            point=pymupdf.Point(line['spans'][0]['origin'])
            if not clip.contains(point): continue
            text='Inside training range' if 'inside' in line['spans'][0]['text'] else 'Outside training range'
            p.insert_text((point.x-clip.x0,point.y-clip.y0),text,fontname='helv',fontsize=9)
        for line in horizontal:
            point=pymupdf.Point(line['spans'][0]['origin'])
            if not clip.contains(point): continue
            text='Station altitude - model orography (m)'
            center=(line['bbox'][0]+line['bbox'][2])/2-clip.x0
            length=pymupdf.get_text_length(text,fontname='helv',fontsize=9)
            p.insert_text((center-length/2,point.y-clip.y0),text,fontname='helv',fontsize=9)
        items=[s for s in spans if clip.contains(pymupdf.Point(s['origin']))]
        rects=[]
        for j,s in enumerate(items):
            ox,oy=s['origin']; x,y=ox-clip.x0,oy-clip.y0
            anchor=pymupdf.Point(x-2,y+2)
            text=str(j+1) if numbered else s['text']
            # Maps use short call-outs with the original full text in the caption.
            if numbered:
                if i==0: placements=[(x+3,y-7),(x-15,y+9),(x+3,y-3)]
                elif i==1: placements=[(x-15,y+10),(x-15,y-8),(x+3,y-3)]
                elif i==2: placements=[(x+20,y+7),(x+15,y-12),(x-20,y+7)]
                else: placements=[(x-17,y+12),(x-18,y),(x-14,y-10)]
                tx,ty=placements[j]
            else:
                tx=x;ty=y
                width=pymupdf.get_text_length(text,fontname='helv',fontsize=9)
                tx=min(max(tx,4),p.rect.width-width-14)
                # Separate labels in the lower-left German cluster.
                if i==0 and j==0: tx=max(65,x-4);ty=y-2
                if i==0 and j==1: tx=x+15;ty=y+1
                if i==3 and j==2: ty=y-5
                if i==3 and j==3: ty=y+8
            width=pymupdf.get_text_length(text,fontname='helv',fontsize=9)
            rect=pymupdf.Rect(tx-1,ty-10,tx+width+1,ty+2)
            # Add a leader only when an annotation has been displaced.
            if abs(tx-x)>3 or abs(ty-y)>3:
                p.draw_line(anchor,pymupdf.Point(tx,ty-3),color=(.45,.45,.45),width=.35)
            p.draw_rect(rect,color=None,fill=(1,1,1),fill_opacity=.9,overlay=True)
            p.insert_text((tx,ty),text,fontname='helv',fontsize=9,color=(0,0,0))
            rects.append(rect)
        result=PdfWriter(clone_from=io.BytesIO(panel.tobytes(garbage=4,deflate=True)))
        result.pdf_header='%PDF-1.5'
        result.write(OUT/f'FigS{start+i}.pdf')
        panel.close()
    original.close(); edited.close()

def main():
    folds();spatial_scores();did();original_three_seed_scores()
    edit_vector_panels(OUT/'fig_station_diff_maps_spatial.pdf',
                       [(0,0,324,306),(324,0,648,306),(0,306,324,612),(324,306,648,612)],2,True)
    edit_vector_panels(OUT/'fig_extrapolation_diagnostic.pdf',
                       [(0,24,370,298),(370,24,720,298),(0,298,370,576),(370,298,720,576)],6)
    print('Rendered Fig1–Fig3 and FigS1–FigS9 using saved numerical and vector assets.')

if __name__=='__main__':
    main()
