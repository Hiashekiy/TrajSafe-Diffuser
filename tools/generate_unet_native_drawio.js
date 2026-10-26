const fs = require('fs');
const path = require('path');

const W = 1400;
const H = 596;
const outDir = path.resolve(__dirname, '..', 'artifacts');
const outPath = path.join(outDir, 'neural_iris_diffuser_unet_native.drawio');
fs.mkdirSync(outDir, { recursive: true });

const C = {
  orange: '#E69500', orangeFill: '#FDE7CA', beige: '#E8D2B5',
  red: '#C94F4F', redFill: '#F3C7C7', blue: '#4E7DBB',
  blueFill: '#C9DCF4', gray: '#555555', grayFill: '#DEDEDE',
};

let seq = 2;
const cells = [];
const esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const fmt = (v) => String(Math.round(v * 1000) / 1000);

function polygon(points, fill, stroke, sw = 1, idPrefix = 'poly') {
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  const w = Math.max(...xs) - x || 1, h = Math.max(...ys) - y || 1;
  const coords = points.map(([px, py]) => [+(fmt((px - x) / w)), +(fmt((py - y) / h))]);
  const id = `${idPrefix}_${seq++}`;
  const style = `shape=mxgraph.basic.polygon;polyCoords=${JSON.stringify(coords)};polyline=0;whiteSpace=wrap;html=1;fillColor=${fill};strokeColor=${stroke};strokeWidth=${sw};shadow=0;`;
  cells.push(`        <mxCell id="${id}" value="" style="${esc(style)}" vertex="1" parent="1"><mxGeometry x="${fmt(x)}" y="${fmt(y)}" width="${fmt(w)}" height="${fmt(h)}" as="geometry"/></mxCell>`);
  return id;
}

function rectangle(x, y, w, h, fill, stroke = 'none', sw = 0, idPrefix = 'rect') {
  return polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], fill, stroke, sw, idPrefix);
}

function label(x, y, w, h, value, align = 'center', size = 14, weight = true) {
  const id = `text_${seq++}`;
  const style = `text;html=1;strokeColor=none;fillColor=none;align=${align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize=${size};fontStyle=${weight ? 1 : 0};`;
  cells.push(`        <mxCell id="${id}" value="${esc(value)}" style="${esc(style)}" vertex="1" parent="1"><mxGeometry x="${fmt(x)}" y="${fmt(y)}" width="${fmt(w)}" height="${fmt(h)}" as="geometry"/></mxCell>`);
}

function hatchRect(x, y, w, h, color, spacing = 14) {
  // Editable diagonal strips, clipped analytically to the rectangle bounds.
  for (let s = -h; s < w; s += spacing) {
    const p1x = Math.max(0, s), p1y = Math.max(0, -s);
    const p2x = Math.min(w, h + s), p2y = Math.min(h, w - s);
    if (p2x <= p1x || p2y <= p1y) continue;
    const t = 0.65;
    polygon([[x+p1x,y+p1y],[x+p1x+t,y+p1y],[x+p2x+t,y+p2y],[x+p2x,y+p2y]], color, 'none', 0, 'hatch');
  }
}

function volume({x, y, t, h, d, fill, stroke, hatch = null}) {
  polygon([[x,y],[x+t,y],[x+t+d,y-d],[x+d,y-d]], '#FFFFFF', stroke, 1, 'top');
  polygon([[x+t,y],[x+t+d,y-d],[x+t+d,y-d+h],[x+t,y+h]], fill, stroke, 1, 'side');
  rectangle(x, y, Math.max(t, 1), h, hatch ? '#FFFFFF' : fill, stroke, 1, 'front');
  if (hatch) hatchRect(x + 0.6, y + 0.6, Math.max(t - 1.2, 0.5), h - 1.2, hatch);
}

function cuboid({x, y, w, h, d = 9, fill, stroke, hatch = null}) {
  polygon([[x,y],[x+d,y-d],[x+w+d,y-d],[x+w,y]], '#FFFFFF', stroke, 1, 'legend_top');
  polygon([[x+w,y],[x+w+d,y-d],[x+w+d,y+h-d],[x+w,y+h]], fill, stroke, 1, 'legend_side');
  rectangle(x, y, w, h, hatch ? '#FFFFFF' : fill, stroke, 1, 'legend_front');
  if (hatch) hatchRect(x + 0.5, y + 0.5, w - 1, h - 1, hatch, 7);
}

function arrow(x1, y1, x2, y2, color = '#000000', width = 2) {
  const head = 12;
  const shaftEnd = x2 - head;
  rectangle(x1, y1 - width / 2, Math.max(1, shaftEnd - x1), width, color, 'none', 0, 'arrow_shaft');
  polygon([[shaftEnd,y2-7],[x2,y2],[shaftEnd,y2+7]], color, 'none', 0, 'arrow_head');
}

// Encoder volumes.
volume({x:13,y:171,t:23,h:362,d:128,fill:'#FFFFFF',stroke:C.gray,hatch:'#9B9B9B'});
volume({x:36,y:171,t:12,h:362,d:128,fill:C.orangeFill,stroke:C.orange});
volume({x:59,y:171,t:12,h:362,d:128,fill:C.beige,stroke:C.orange});
label(150,17,70,24,'conv1');

volume({x:105,y:353,t:12,h:199,d:82,fill:C.redFill,stroke:C.red});
volume({x:129,y:353,t:28,h:199,d:82,fill:C.orangeFill,stroke:C.orange});
volume({x:164,y:353,t:27,h:199,d:82,fill:C.beige,stroke:C.orange});
label(202,243,60,24,'conv2');

volume({x:210,y:460,t:25,h:110,d:46,fill:C.redFill,stroke:C.red});
volume({x:245,y:460,t:48,h:110,d:46,fill:C.orangeFill,stroke:C.orange});
volume({x:304,y:460,t:47,h:110,d:47,fill:C.beige,stroke:C.orange});
label(300,386,60,24,'conv3');

volume({x:362,y:521,t:47,h:62,d:23,fill:C.redFill,stroke:C.red});
volume({x:420,y:521,t:70,h:62,d:23,fill:C.orangeFill,stroke:C.orange});
volume({x:502,y:521,t:70,h:62,d:23,fill:C.orangeFill,stroke:C.orange});
label(458,469,60,24,'conv4');

// Decoder volumes.
volume({x:595,y:460,t:47,h:110,d:46,fill:'#FFFFFF',stroke:C.orange,hatch:'#F0B54A'});
volume({x:642,y:460,t:47,h:110,d:46,fill:C.blueFill,stroke:C.blue});
volume({x:712,y:460,t:47,h:110,d:46,fill:C.orangeFill,stroke:C.orange});
label(676,385,68,24,'deconv1');

volume({x:782,y:353,t:24,h:199,d:82,fill:'#FFFFFF',stroke:C.orange,hatch:'#F0B54A'});
volume({x:806,y:353,t:28,h:199,d:82,fill:C.blueFill,stroke:C.blue});
volume({x:851,y:353,t:28,h:199,d:82,fill:C.orangeFill,stroke:C.orange});
label(860,243,70,24,'deconv2');

volume({x:933,y:171,t:12,h:362,d:128,fill:'#FFFFFF',stroke:C.orange,hatch:'#F0B54A'});
volume({x:945,y:171,t:16,h:362,d:128,fill:C.blueFill,stroke:C.blue});
volume({x:972,y:171,t:14,h:362,d:128,fill:C.beige,stroke:C.orange});
label(1048,17,76,24,'deconv3');
volume({x:1014,y:171,t:1,h:362,d:128,fill:C.grayFill,stroke:C.gray});

// Skip connections as fully editable vertex geometry (no dangling edges).
arrow(164,194,929,196);
arrow(268,364,778,364);
arrow(385,438,620,438);

// Legend.
const lx = 1178, tx = 1227;
cuboid({x:lx,y:59,w:25,h:23,d:10,fill:'#FFFFFF',stroke:C.gray,hatch:'#9B9B9B'});
label(tx,49,130,28,'input','left');
cuboid({x:lx,y:97,w:25,h:17,d:8,fill:C.orangeFill,stroke:C.orange});
label(tx,89,150,28,'convolution','left');
cuboid({x:lx,y:133,w:25,h:17,d:8,fill:C.redFill,stroke:C.red});
label(tx,125,150,28,'max pooling','left');
cuboid({x:lx,y:169,w:25,h:17,d:8,fill:'#FFFFFF',stroke:C.orange,hatch:'#F0B54A'});
label(tx,163,150,28,'concatenate','left');
cuboid({x:lx,y:206,w:25,h:16,d:8,fill:C.blueFill,stroke:C.blue});
label(tx,199,150,28,'deconvolution','left');
arrow(lx,246,lx+34,246);
label(tx,233,160,28,'skip connection','left');
cuboid({x:lx+1,y:275,w:25,h:22,d:9,fill:C.grayFill,stroke:C.gray});
label(tx,272,150,28,'segmentation','left');

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-09-30T00:00:00.000Z" agent="Codex" version="28.2.5" type="device">
  <diagram id="unet-native" name="U-Net Native Editable">
    <mxGraphModel dx="1400" dy="596" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" pageHeight="596" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
${cells.join('\n')}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
`;

fs.writeFileSync(outPath, xml, 'utf8');
console.log(`${outPath}\nvertices=${cells.length}\nimages=0`);
