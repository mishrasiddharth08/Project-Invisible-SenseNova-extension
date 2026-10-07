from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
BG, CARD, INK, MUTED, BLUE, GREEN = '#0b1220', '#152237', '#f1f5f9', '#a4b5cc', '#84bcff', '#72e2ba'

class Visual:
    def __init__(self, title, description, height=900):
        self.height = height
        self.image = Image.new('RGB', (2800, height*2), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc><rect width="1400" height="{height}" fill="{BG}"/>']
    def rect(self, x, y, w, h, color=CARD, radius=18):
        self.draw.rounded_rectangle((x*2,y*2,(x+w)*2,(y+h)*2), radius=radius*2, fill=color)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{color}"/>')
    def text(self, x, y, value, size=24, color=INK, bold=False):
        try:
            font = ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'), size*2)
        except OSError:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size*2)
        self.draw.text((x*2,y*2), value, font=font, fill=color)
        self.svg.append(f'<text x="{x}" y="{y+size}" fill="{color}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{escape(value)}</text>')
    def save(self, name):
        self.image.resize((1400,self.height), Image.Resampling.LANCZOS).save(OUT/(name+'.png'), optimize=True)
        (OUT/(name+'.svg')).write_text('\n'.join(self.svg+['</svg>'])+'\n', encoding='utf-8', newline='\n')

v = Visual('Project Invisible: SenseNova and Looped-DiT', 'Model comparison, workflow and validation for the native Forge Neo extension.')
v.text(60,35,'PROJECT INVISIBLE',22,GREEN,True)
v.text(60,75,'Two models. One familiar Forge workflow.',46,INK,True)
v.text(60,139,'Choose a checkpoint → write a prompt → press the native Generate button.',25,MUTED)
for x,title,color in ((60,'SenseNova U1.5',BLUE),(730,'Looped-DiT B16 / B32',GREEN)):
    v.rect(x,205,610,280)
    v.rect(x,205,610,6,color,2)
    v.text(x+26,228,title,32,color,True)
v.text(86,290,'Create images or edit a source in img2img.',24)
v.text(86,338,'Base model: try 1024×1024 · 50 steps · CFG 4.',22,MUTED)
v.text(86,382,'Official 8-step adapter: text-to-image only.',22,MUTED)
v.text(86,426,'Editing is experimental; masks are unsupported.',21,MUTED)
v.text(756,290,'Text-to-image with shared transformer loops.',24)
v.text(756,338,'512×512 · 100 Euler steps · CFG 6 · 4 loops.',22,MUTED)
v.text(756,382,'Needs its matching local FLAN-T5-Large encoder.',22,MUTED)
v.text(756,426,'Fewer loops reduce work and change quality.',21,MUTED)
v.text(60,530,'GET STARTED',21,GREEN,True)
steps=[('1','Get models','Manual or automatic'),('2','Refresh dropdown','Select your checkpoint'),('3','Choose settings','Use its settings button'),('4','Generate','Watch the image develop'),('5','Save / switch','Native gallery + unload')]
for i,(number,title,detail) in enumerate(steps):
    x=60+i*260
    v.rect(x,580,240,126)
    v.text(x+18,593,number,28,GREEN,True)
    v.text(x+18,638,title,22,INK,True)
    v.text(x+18,676,detail,16,MUTED)
v.rect(60,750,1280,80,'#18332e')
v.text(86,764,'Current + overall progress   ·   Final image matches saved output   ·   Memory offloading',24,GREEN,True)
v.text(60,855,'Oct 6 validation: 62 tests, including small-model CUDA parity. Full weights need visual validation.',20,MUTED)
v.save('integration-overview')

def ui_guide(looped):
    label='Looped-DiT B16 / B32' if looped else 'SenseNova U1.5'
    v=Visual(label+' UI guide','Illustrated current Forge controls; not a screenshot.',1100)
    v.text(60,30,'PROJECT INVISIBLE / CURRENT UI GUIDE',22,GREEN,True)
    v.text(60,75,label+' in the native Forge workflow',42,INK,True)
    v.text(60,134,'Illustrated guide, not a live screenshot. Control labels match the extension; placement may vary.',21,MUTED)
    v.rect(60,190,1280,86)
    v.text(82,203,'Native checkpoint dropdown',18,MUTED)
    v.text(82,235,'Looped-DiT / looped-dit-b16' if looped else 'Your SenseNova U1.5 checkpoint',25,BLUE,True)
    v.text(874,232,'Refresh',21,MUTED)
    v.rect(1110,209,204,47,'#235239',10)
    v.text(1142,215,'Generate',25,GREEN,True)
    v.rect(60,299,745,702)
    v.rect(82,318,126,42,'#254665',8)
    v.text(105,324,'txt2img',23,BLUE,True)
    v.text(237,324,'img2img / U1.5 editing only',21,MUTED)
    v.text(82,380,'Prompt',18,MUTED)
    v.rect(82,414,699,74,'#0d1829',8)
    v.text(98,430,'Describe your image, or the edit in img2img.',23)
    v.text(82,515,'512 x 512 / 100 steps / CFG 6' if looped else '1024 x 1024 / 50 steps / CFG 4',23,BLUE,True)
    v.text(82,554,'SenseNova - Project Invisible',24,INK,True)
    v.text(82,593,'Natural photo details: off with Looped-DiT preset' if looped else 'Natural photo details: optional prompt instructions',21,MUTED)
    v.rect(82,635,699,48,'#235239',8)
    v.text(100,645,'Use Looped-DiT settings (inside Advanced)' if looped else 'Use photo quality settings / 50 steps / CFG 4',22,GREEN,True)
    v.text(82,707,'Advanced / optional controls',23,INK,True)
    rows=['Looped-DiT depth: 4 (1-16 available)','Local resources: matching FLAN-T5-Large folder','Memory mode: Auto / Full / offload options','U1.5 fast adapter and Think must be disabled'] if looped else ['DeGrid / off by default     Memory mode / Auto','8-step speed adapter / txt2img only (local LoRA)','Local resources folder / U1.5 config and tokenizer','Think / optional     Timestep shift / default 3']
    for i,row in enumerate(rows): v.text(82,751+i*40,row,20,MUTED)
    v.text(82,938,'Unload SenseNova / releases either engine',21,BLUE)
    v.rect(828,299,512,462)
    v.text(850,320,'Native live preview / gallery',23,INK,True)
    v.rect(850,365,468,210,'#243447',10)
    v.text(907,441,'Image develops here',25,MUTED,True)
    v.text(923,483,'Illustrated preview area',19,MUTED)
    v.text(850,605,'Current image / illustrative 32%',19,BLUE)
    v.rect(850,641,468,14,'#0d1829',6)
    v.rect(850,641,150,14,BLUE,6)
    v.text(850,680,'Overall generation / illustrative 8%',19,GREEN)
    v.rect(850,716,468,14,'#0d1829',6)
    v.rect(850,716,38,14,GREEN,6)
    v.rect(828,787,512,214)
    v.text(850,803,'Get models / manual or automatic',22,INK,True)
    for i,row in enumerate(['1. Select a model from the list','2. Manual: follow the download guide','3. Automatic: Download selected model','4. Refresh the checkpoint dropdown']):
        v.text(850,847+i*36,row,20,MUTED)
    v.text(60,1035,'U1.5 editing is experimental. Looped-DiT: text-to-image only. Hires fix and masks are unsupported.',20,MUTED)
    v.save('ui-guide' if looped else 'ui-guide-u15')
ui_guide(False)
ui_guide(True)
