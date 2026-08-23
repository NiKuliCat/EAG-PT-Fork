# this script is automatically executed in setup.py along with `pip install`
nvcc -ptx -arch=compute_89 devicePrograms.cu -o ../ptx/devicePrograms.ptx -I../external/optix/
bin2c ../ptx/devicePrograms.ptx --const --name devicePrograms_ptx > ../ptx/devicePrograms.h
