const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const pngPath = path.join(root, 'artifacts', 'neural_iris_diffuser_unet_preview.png');
const outPath = path.join(root, 'artifacts', 'neural_iris_diffuser_unet_v2.drawio');

if (!fs.existsSync(pngPath)) {
  throw new Error(`Missing rendered PNG: ${pngPath}`);
}

// draw.io's style parser uses semicolons as separators. Per the repository's
// HOWTO-drawio.md, embedded PNGs must omit the conventional ";base64" marker.
const pngBase64 = fs.readFileSync(pngPath).toString('base64');

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2026-09-30T00:00:00.000Z" agent="Codex" version="28.2.5" type="device">
  <diagram id="unet-architecture-v2" name="U-Net Architecture">
    <mxGraphModel dx="1400" dy="596" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1400" pageHeight="596" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="unet_image" value="" style="shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,${pngBase64};" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="1400" height="596" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
`;

// Write a new file rather than overwriting the document currently open in draw.io.
fs.writeFileSync(outPath, xml, { encoding: 'utf8' });

// Immediate structural checks required by the handoff instructions.
if (!xml.includes('image=data:image/png,') || xml.includes('image=data:image/png;base64,')) {
  throw new Error('Embedded image URI does not follow HOWTO-drawio.md');
}
if (!xml.includes('id="unet_image" value=""')) {
  throw new Error('Image cell must have an empty value');
}

console.log(outPath);
