-------------------------------------------------------------------------------
# Dense RGB-D Mapping and Relocalisation                            
-------------------------------------------------------------------------------

## Introduction

This is an open source C++ implementation of [1]. It provides scene mapping and 
fast and robust wide baseline relocalisation without the need for training. The 
code has been tested on both Ubuntu and Windows platforms.

[1] RGB-D Relocalisation Using Pairwise Geometry and Concise Key Point Sets. 
Li, S., & Calway, A. ICRA, 2015. 

Please consider citing above publication if you use this software in your own 
work.

The repository is maintained by Shuda Li (csxsl@bristol.ac.uk). Feel free to contact
me, if you have any question or suggestions.

-------------------------------------------------------------------------------

## Functional keys

'7'  switch on the dense surface reconstructed.

'0', pressing 0 key once aligns the viewing position of up left sub window with current camera pose only.
     pressing 0 key one more time locks the viewing position with the camera location and pressing 0 again release the lock. 
	 
'F5', switch on/off the recorded camera path (red/blue curves)

'M', convert the TSDF surface representation into a triangle mesh, to display it, press '7'  

More key settings can be found in Viewer::keyPressEvent() in the ./live_mapping_and_relocalisation/MultiViewer.cpp 

-------------------------------------------------------------------------------

## License

The source code is released under the MIT license. In short, you can do 
anything with the code for any purposes. For details please view the license file 
in the source codes.

-------------------------------------------------------------------------------

## Dependences

- OpenCV 3.0 (BSD license)
- Eigen (GNU free software)
- Sopuhs (GNU free software)
- OpenNI 2 (Apache license)
- Boost (BSD License)
- GLEW (Free commercial use)
- NIFTI (Public domain)
- ZLIB (Free commercial use)
- CUDA

-------------------------------------------------------------------------------

## Compilation

- Windows (7/8/8.1/10)
  https://docs.google.com/document/d/1-THFy0CCAK2jgCwKX78hUfOQn2L0bInoNJH9JiziH18/edit?usp=sharing
- Ubuntu (14.04/15.04)
  https://docs.google.com/document/d/13TT0n8cH1gFA_FZZOV2TD97_VfGK10WUeM1pSlsOQ3o/edit?usp=sharing
  
-------------------------------------------------------------------------------

## Demo Video and talk

- https://lishuda.wordpress.com/
- https://youtu.be/TUd86IXQOrA
- https://youtu.be/gAcEhu5V6-I
- https://youtu.be/xkFhLtLr5rg  

-------------------------------------------------------------------------------

## Project page
- https://lishuda.wordpress.com/