# 3DHISTECH SlideViewer ScriptQuant

The proposed method for mitosis detection using GaussUNet is implemented on SlideVeiwer using ScriptQuant.

## Software version

SlideViewer 2.5 (64-bit version) for Windows (Free to use)

https://www.3dhistech.com/research/software-downloads/

QuantCenter 2.3 (64-bit version) for Windows (License required)

https://www.3dhistech.com/research/quantcenter/


## How to use

1. Place this scriptquant directory under Program Files/3DHISTECH/QuantCenter.

2. Launch SlideVeiwer and select WSI.

3. Launch ScriptQuant from the QuantCenter menu.

4. Select scriptquant/gauss_unet.py from File>import.

5. Set the detection threshold between 0~255 (default: 128).

6. Press the Next button and then press Start measurement.

7. Inference begins and the prediction result is marked with a green square.