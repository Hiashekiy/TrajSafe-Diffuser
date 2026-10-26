"""Insert supplied figures into the existing IEEE manuscript, without changing results."""
from pathlib import Path
import json, re, shutil
import pymupdf as fitz
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parent
PICTURES=ROOT.parent/'picture'
DEST=ROOT/'figures'/'prepared'; DEST.mkdir(parents=True,exist_ok=True)
BACKUP=ROOT/'backups';BACKUP.mkdir(exist_ok=True)
source_backup=BACKUP/'main_before_figure_layout.tex'
if not source_backup.exists():shutil.copy2(ROOT/'main.tex',source_backup)
if (ROOT/'main.pdf').exists() and not (BACKUP/'main_before_figure_layout.pdf').exists():shutil.copy2(ROOT/'main.pdf',BACKUP/'main_before_figure_layout.pdf')

mapping=[]
def prepare_image(original,target):
    im=Image.open(PICTURES/original).convert('RGB')
    # Only exterior white margins are removed; all plotted content is kept.
    diff=ImageChops.difference(im,Image.new('RGB',im.size,'white')).convert('L')
    bbox=diff.point(lambda p:255 if p>12 else 0).getbbox()
    if bbox:
        a,b,c,d=bbox;pad=8
        bbox=(max(0,a-pad),max(0,b-pad),min(im.width,c+pad),min(im.height,d+pad))
        im=im.crop(bbox)
    im.save(DEST/target,optimize=True)
    mapping.append({'source':str(PICTURES/original),'prepared':str(DEST/target),'exterior_crop':bbox,'size':list(im.size)})

for original,target in [
 ('总体框架图.drawio.png','fig01_framework.png'),
 ('安全约束生成.png','fig02_safety_generation.png'),
 ('alm_continuous_safety.png','fig03_alm.png'),
 ('E2_fig1_feedback_process.png','fig04_feedback.png'),
 ('E3_fig1_alm_budget.png','fig05_alm_budget.png'),
]:prepare_image(original,target)

# One self-contained CARLA figure: four scene types, two examples per type.
# Preserve each square image in full; no scene, trajectory or obstacle is cropped.
pdf=fitz.open();width=516.;gutter=6.;panel=(width-3*gutter)/4;header=17.;row_label=15.;row_gap=5.
height=header+2*(panel+row_label)+row_gap
page=pdf.new_page(width=width,height=height)
fontfile='C:/Windows/Fonts/times.ttf';page.insert_font(fontname='TimesEmbedded',fontfile=fontfile)
categories=[('Straight road','直行'),('Intersection','拐弯'),('Curved road','弯道'),('Long-distance route','长距离')]
carla=[]
for col,(title,name) in enumerate(categories):
    x=col*(panel+gutter)
    page.insert_textbox(fitz.Rect(x,0,x+panel,header),title,fontname='TimesEmbedded',fontsize=9,align=1,color=(.12,.15,.18))
    for row in range(2):
        y=header+row*(panel+row_label+row_gap)
        image=PICTURES/'carla'/f'{name}{row+1}.png'
        rect=fitz.Rect(x,y,x+panel,y+panel)
        page.insert_image(rect,filename=str(image),keep_proportion=True)
        letter=chr(ord('a')+row*4+col)
        page.insert_textbox(fitz.Rect(x,y+panel+1,x+panel,y+panel+row_label),f'({letter})',fontname='TimesEmbedded',fontsize=8.8,align=1,color=(.12,.15,.18))
        carla.append({'source':str(image),'panel':letter,'category':title,'row':row+1,'rect':list(rect)})
pdf.save(DEST/'fig06_carla.pdf',garbage=4,deflate=True)
page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(DEST/'fig06_carla.png')
pdf.close()

def figure(label,filename,caption,wide=True,scale=1,position=None):
    env='figure*' if wide else 'figure'
    dimension=(r'\textwidth' if wide else r'\columnwidth')
    width=dimension if scale==1 else str(scale)+dimension
    position=position or ('!tb' if wide else '!htb')
    return '\n'+r'\begin{'+env+'}['+position+']\n'+r'\centering'+'\n'+r'\includegraphics[width='+width+']{'+filename+'}\n'+r'\caption{'+caption+'}\n'+r'\label{fig:'+label+'}\n'+r'\end{'+env+'}\n\n'

text=source_backup.read_text(encoding='utf-8')
text=text.replace('%  本版不含图片（§4.3.2 与 §4.5 的图题留在中文源稿，见文末注释）。','%  Supplied method and experiment figures inserted; CARLA examples combined into one figure.')
text=text.replace('%                  (2 source figure captions kept as comments at the end)','%                  (supplied figures restored with in-text references)')
packages=r'''\usepackage{bm}
\usepackage{graphicx}
\usepackage{dblfloatfix}
\usepackage[nospread]{flushend}
\graphicspath{{figures/prepared/}}

% Compact IEEE two-column floats; retain the class fonts and page geometry.
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 1pt}
\setlength{\dblfloatsep}{7pt plus 2pt minus 1pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{0pt}
\renewcommand{\topfraction}{0.93}
\renewcommand{\bottomfraction}{0.88}
\renewcommand{\dbltopfraction}{0.93}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{7}
\setcounter{dbltopnumber}{3}'''
assert text.count(r'\usepackage{bm}')==1
text=text.replace(r'\usepackage{bm}',packages)

framework=figure('framework','fig01_framework.png','Overall framework of TrajSafe-Diffuser. Feature encoding and fusion condition safety geometry generation and trajectory denoising. ALM correction updates the sampling state and supplies historical safety feedback.')
safety=figure('safety','fig02_safety_generation.png',r'Local safety feature extraction and ellipse prediction. Selected-route features query local map features through time-conditioned cross-attention with a spatial distance bias. The predicted ellipses initialize the subsequent convex-region construction.')
alm=figure('alm','fig03_alm.png',r'Continuous-trajectory safety constraints and ALM correction. Region responsibility boundaries and B-spline knots define cubic B\'ezier subsegments. Constraining their four control points to the assigned convex region gives continuous segment containment; ALM adjusts the spline control points while fixing the endpoints.',scale=.8)
feedback=figure('feedback','fig04_feedback.png',r'Effect of historical safety feedback on the denoising process: (a) mean safety violation before ALM correction; (b) mean ALM correction magnitude.',wide=False)
budget=figure('alm-budget','fig05_alm_budget.png',r'ALM iteration budget analysis: (a) continuous-constraint satisfaction and collision rate; (b) end-to-end inference cost. The default configuration is marked by a star.',wide=False)
carla_fig=figure('carla','fig06_carla.pdf',r'Trajectory generation in CARLA road scenes. Columns show straight roads, intersections, curved roads and long-distance routes; the two rows show separate examples. Green bands indicate the drivable range and highlighted centerlines show the generated trajectories.',scale=.82,position='!t')

text=text.replace('The overall structure of TrajSafe-Diffuser comprises three modules:',r'The overall structure of TrajSafe-Diffuser, shown in Fig.~\ref{fig:framework}, comprises three modules:')
text=text.replace(r'\subsection{Feature Encoding and Fusion}',framework+r'\subsection{Feature Encoding and Fusion}',1)
text=text.replace(r'\subsection{Generative Safety Constraint Construction}',r'\subsection{Generative Safety Constraint Construction}'+'\n'+safety,1)
text=text.replace('With the features $H_{S,*}$ of the selected route',r'As illustrated in Fig.~\ref{fig:safety}, with the features $H_{S,*}$ of the selected route',1)
text=text.replace(r'\subsubsection{Continuous-Trajectory Safety Constraint and ALM Correction}',r'\subsubsection{Continuous-Trajectory Safety Constraint and ALM Correction}'+'\n'+alm,1)
text=text.replace('The augmented Lagrangian method (ALM) is used to apply safety correction',r'As shown in Fig.~\ref{fig:alm}, the augmented Lagrangian method (ALM) is used to apply safety correction',1)

old_feedback='% 源稿图题（本版不带图，图题保留在中文源稿）：\n% 图 1 历史安全反馈对去噪过程的影响。(a) ALM 修正前的平均安全违规量；\n% (b) 平均 ALM 修正幅度。'
assert old_feedback in text
text=text.replace(old_feedback,r'Figure~\ref{fig:feedback} shows the corresponding denoising-process curves.'+'\n'+feedback)
text=text.replace(r'\subsection{ALM Iteration Budget Analysis}',r'\subsection{ALM Iteration Budget Analysis}'+'\n'+budget,1)
text=text.replace('To assess the influence of the maximum number of ALM inner iterations on safety,',r'Figure~\ref{fig:alm-budget} visualizes the safety--budget relation and inference cost. To assess the influence of the maximum number of ALM inner iterations on safety,',1)
text=text.replace('three typical scenes, namely a straight road segment, an intersection and a\ncurved road segment, are selected in CARLA;','four scene types, namely straight road segments, intersections, curved road\nsegments and long-distance routes, are selected in CARLA (Fig.~\\ref{fig:carla});',1)
old_carla='% 源稿图题（本版不带图，图题保留在中文源稿）：\n% 图 4-6 CARLA 典型道路场景下的轨迹生成结果。（a）直道路段；（b）交叉路口；\n% （c）弯道路段。'
assert old_carla in text
text=text.replace(old_carla,'')
text=text.replace(budget,budget+carla_fig,1)
assert text.count(r'\begin{figure*}')==4
assert text.count(r'\begin{figure}')==2
(ROOT/'main.tex').write_text(text,encoding='utf-8')
(DEST/'image_mapping.json').write_text(json.dumps({'method_and_experiment_figures':mapping,'carla_panels':carla},ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated main.tex; prepared five supplied figures and one 2x4 CARLA composite.')
