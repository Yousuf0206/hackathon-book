const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function convert() {
  const projectRoot = path.resolve(__dirname, '..');
  const imgDir = path.join(projectRoot, 'static', 'img');

  const tasks = [
    {
      src: path.join(imgDir, 'logo.svg'),
      dest: path.join(imgDir, 'logo.jpg'),
      width: 1200,
      height: 320,
      quality: 90,
    },
    {
      src: path.join(imgDir, 'physical-ai-humanoid-robotics.svg'),
      dest: path.join(imgDir, 'physical-ai-humanoid-robotics.jpg'),
      width: 3840,
      height: 1920,
      quality: 92,
    },
  ];

  // Allow converting only a specific asset via command-line arg: 'hero' or 'logo'
  const arg = process.argv[2];
  if (arg === 'hero') {
    tasks.splice(0, tasks.length, tasks.find(t => t.src.endsWith('physical-ai-humanoid-robotics.svg')));
  } else if (arg === 'logo') {
    tasks.splice(0, tasks.length, tasks.find(t => t.src.endsWith('logo.svg')));
  }

  for (const t of tasks) {
    if (!fs.existsSync(t.src)) {
      console.error('Source not found:', t.src);
      continue;
    }
    console.log('Converting', path.basename(t.src), '→', path.basename(t.dest));
    try {
      const svgBuffer = fs.readFileSync(t.src);
      await sharp(svgBuffer)
        .resize(t.width, t.height, { fit: 'cover' })
        .flatten({ background: '#ffffff' })
        .jpeg({ quality: t.quality })
        .toFile(t.dest);
      console.log('Wrote', t.dest);
    } catch (err) {
      console.error('Error converting', t.src, err);
    }
  }
}

convert().catch((e) => {
  console.error(e);
  process.exit(1);
});
