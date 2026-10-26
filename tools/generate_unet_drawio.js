const fs = require('fs');
const path = require('path');

const W = 1400;
const H = 596;
const outDir = path.resolve(__dirname, '..', 'artifacts');
fs.mkdirSync(outDir, { recursive: true });

const C = {
  orange: '#E69500',
  orangeFill: '#FDE7CA',
  beige: '#E8D2B5',
  red: '#C94F4F',
  redFill: '#F3C7C7',
  blue: '#4E7DBB',
  blueFill: '#C9DCF4',
  gray: '#555555',
  grayFill: '#DEDEDE',
};

const esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const n = (v) => Math.round(v * 10) / 10;
const pts = (a) => a.map(([x, y]) => `${n(x)},${n(y)}`).join(' ');

let body = '';
const add = (s) => { body += s + '\n'; };

function polygon(points, fill, stroke, sw = 1) {
  add(`<polygon points="${pts(points)}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
}

function line(x1, y1, x2, y2, stroke = '#000', sw = 2, marker = '') {
  add(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${sw}"${marker ? ` marker-end="url(#${marker})"` : ''}/>`);
}

function text(x, y, value, anchor = 'middle', size = 14, weight = 700) {
  add(`<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" font-weight="${weight}" fill="#000">${esc(value)}</text>`);
}

// A CNN feature volume: narrow front face with a receding top and a broad side plane.
function volume({x, y, t, h, d, fill, stroke, pattern, id}) {
  const faceFill = pattern ? `url(#${pattern})` : fill;
  add(`<g${id ? ` id="${id}"` : ''}>`);
  polygon([[x,y],[x+t,y],[x+t+d,y-d],[x+d,y-d]], '#FFFFFF', stroke, 1);
  polygon([[x+t,y],[x+t+d,y-d],[x+t+d,y-d+h],[x+t,y+h]], fill, stroke, 1);
  polygon([[x,y],[x+t,y],[x+t,y+h],[x,y+h]], faceFill, stroke, 1);
  add(`</g>`);
}

function cuboid({x,y,w,h,d=10,fill,stroke,pattern}) {
  const frontFill = pattern ? `url(#${pattern})` : fill;
  polygon([[x,y],[x+d,y-d],[x+w+d,y-d],[x+w,y]], '#FFFFFF', stroke, 1);
  polygon([[x+w,y],[x+w+d,y-d],[x+w+d,y+h-d],[x+w,y+h]], fill, stroke, 1);
  polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], frontFill, stroke, 1);
}

// White background keeps export identical in light and dark viewers.
add(`<rect width="${W}" height="${H}" fill="#FFFFFF"/>`);

// Encoder.
volume({x:13,y:171,t:23,h:362,d:128,fill:'#FFFFFF',stroke:C.gray,pattern:'hatchGray'});
volume({x:36,y:171,t:12,h:362,d:128,fill:C.orangeFill,stroke:C.orange});
volume({x:59,y:171,t:12,h:362,d:128,fill:C.beige,stroke:C.orange});
text(185,33,'conv1');

volume({x:105,y:353,t:12,h:199,d:82,fill:C.redFill,stroke:C.red});
volume({x:129,y:353,t:28,h:199,d:82,fill:C.orangeFill,stroke:C.orange});
volume({x:164,y:353,t:27,h:199,d:82,fill:C.beige,stroke:C.orange});
text(232,259,'conv2');

volume({x:210,y:460,t:25,h:110,d:46,fill:C.redFill,stroke:C.red});
volume({x:245,y:460,t:48,h:110,d:46,fill:C.orangeFill,stroke:C.orange});
volume({x:304,y:460,t:47,h:110,d:47,fill:C.beige,stroke:C.orange});
text(330,402,'conv3');

volume({x:362,y:521,t:47,h:62,d:23,fill:C.redFill,stroke:C.red});
volume({x:420,y:521,t:70,h:62,d:23,fill:C.orangeFill,stroke:C.orange});
volume({x:502,y:521,t:70,h:62,d:23,fill:C.orangeFill,stroke:C.orange});
text(488,485,'conv4');

// Decoder and concatenation volumes.
volume({x:595,y:460,t:47,h:110,d:46,fill:'#FFFFFF',stroke:C.orange,pattern:'hatchOrange'});
volume({x:642,y:460,t:47,h:110,d:46,fill:C.blueFill,stroke:C.blue});
volume({x:712,y:460,t:47,h:110,d:46,fill:C.orangeFill,stroke:C.orange});
text(710,401,'deconv1');

volume({x:782,y:353,t:24,h:199,d:82,fill:'#FFFFFF',stroke:C.orange,pattern:'hatchOrange'});
volume({x:806,y:353,t:28,h:199,d:82,fill:C.blueFill,stroke:C.blue});
volume({x:851,y:353,t:28,h:199,d:82,fill:C.orangeFill,stroke:C.orange});
text(895,259,'deconv2');

volume({x:933,y:171,t:12,h:362,d:128,fill:'#FFFFFF',stroke:C.orange,pattern:'hatchOrange'});
volume({x:945,y:171,t:16,h:362,d:128,fill:C.blueFill,stroke:C.blue});
volume({x:972,y:171,t:14,h:362,d:128,fill:C.beige,stroke:C.orange});
text(1085,33,'deconv3');

// Segmentation output plane.
volume({x:1014,y:171,t:0,h:362,d:129,fill:C.grayFill,stroke:C.gray});

// Skip connections, routed to avoid labels.
line(164,194,929,196,'#000',2,'arrow');
line(268,364,778,364,'#000',2,'arrow');
line(385,438,620,438,'#000',2,'arrow');

// Legend.
const lx = 1178, tx = 1227;
cuboid({x:lx,y:59,w:25,h:23,d:10,fill:'#FFFFFF',stroke:C.gray,pattern:'hatchGray'});
text(tx,66,'input','start',14,700);
cuboid({x:lx,y:97,w:25,h:17,d:8,fill:C.orangeFill,stroke:C.orange});
text(tx,106,'convolution','start',14,700);
cuboid({x:lx,y:133,w:25,h:17,d:8,fill:C.redFill,stroke:C.red});
text(tx,141,'max pooling','start',14,700);
cuboid({x:lx,y:169,w:25,h:17,d:8,fill:'#FFFFFF',stroke:C.orange,pattern:'hatchOrange'});
text(tx,181,'concatenate','start',14,700);
cuboid({x:lx,y:206,w:25,h:16,d:8,fill:C.blueFill,stroke:C.blue});
text(tx,216,'deconvolution','start',14,700);
line(lx,246,lx+34,246,'#000',2,'arrow');
text(tx,250,'skip connection','start',14,700);
cuboid({x:lx+1,y:275,w:25,h:22,d:9,fill:C.grayFill,stroke:C.gray});
text(tx,288,'segmentation','start',14,700);

const defs = `
  <defs>
    <pattern id="hatchGray" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(135)">
      <rect width="10" height="10" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="10" stroke="#9B9B9B" stroke-width="1"/>
    </pattern>
    <pattern id="hatchOrange" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(135)">
      <rect width="10" height="10" fill="#FFFFFF"/><line x1="0" y1="0" x2="0" y2="10" stroke="#F0B54A" stroke-width="1"/>
    </pattern>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/>
    </marker>
  </defs>`;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
${defs}
${body}</svg>\n`;

const svgPath = path.join(outDir, 'neural_iris_diffuser_unet.svg');
fs.writeFileSync(svgPath, svg, 'utf8');

// mxGraph styles use semicolons as field separators. A base64 data URI contains
// its own semicolon ("image/svg+xml;base64"), which some draw.io builds split
// prematurely and consequently render as a blank image. URI-encoding the SVG
// avoids that parser ambiguity and is portable across desktop/web draw.io.
const encoded = encodeURIComponent(svg);
const mx = `<mxfile host="app.diagrams.net" modified="2026-09-30T00:00:00.000Z" agent="Codex" version="28.2.5" type="device">
  <diagram id="unet-architecture" name="U-Net Architecture">
    <mxGraphModel dx="1400" dy="596" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" pageHeight="596" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="2" value="" style="shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/svg+xml,${encoded};" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="1400" height="596" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>\n`;

const drawioPath = path.join(outDir, 'neural_iris_diffuser_unet.drawio');
fs.writeFileSync(drawioPath, mx, 'utf8');
console.log(svgPath);
console.log(drawioPath);
