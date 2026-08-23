from pathlib import Path
import numpy as np
import OpenImageIO as oiio

path = Path('Blender/Blender/livingroom/Normal-exr/0000-0001.exr')
inp = oiio.ImageInput.open(str(path))
spec = inp.spec()
pixels = np.asarray(inp.read_image(format=oiio.BASETYPE.FLOAT)).reshape(spec.height, spec.width, spec.nchannels)
inp.close()
print('shape=', pixels.shape, 'channels=', spec.nchannels)
for i in range(min(4, pixels.shape[2])):
    ch = pixels[..., i]
    print(i, 'min=', float(np.nanmin(ch)), 'max=', float(np.nanmax(ch)), 'mean=', float(np.nanmean(ch)))
print('first=', pixels.reshape(-1, pixels.shape[-1])[:8])
print('norm range=', float(np.nanmin(np.linalg.norm(pixels[..., :3], axis=-1))), float(np.nanmax(np.linalg.norm(pixels[..., :3], axis=-1))))
