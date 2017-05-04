//Copyright(c) 2015 Shuda Li[lishuda1980@gmail.com]
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//FOR A PARTICULAR PURPOSE AND NON - INFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
//COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.


#define _USE_MATH_DEFINES
#define EXPORT
#define INFO
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <se3.hpp>

#include "OtherUtil.hpp"
#include "Kinect.h"
#include "GLUtil.hpp"
#include "CudaLib.cuh"

namespace btl{	namespace gl_util
{
	using namespace Eigen;
	using namespace std;
	
	CGLUtil::CGLUtil(ushort uResolution_, ushort uPyrLevel_,btl::utility::tp_coordinate_convention eConvention_ /*= btl::utility::BTL_GL*/,const Eigen::Vector3f& eivCentroid_ /*= Eigen::Vector3f(1.5f,1.5f,0.3f)*/)
		:_uResolution(uResolution_),_usPyrHeight(uPyrLevel_),_usLevel(0),_eConvention(eConvention_),_eivCentroid(eivCentroid_){
			_dZoom = 0.; //for zooming
			_dZoomLast = 1.;
			_dScale = .1;

			_dXAngle = 0;//for rotation.
			_dYAngle = 0;//for rotation.
			_dXLastAngle = 0;
			_dYLastAngle = 0;
			_dX = 0;
			_dY = 0;
			_dXLast = 0;
			_dYLast = 0;

			_nXMotion = 0;
			_nYMotion = 0;

			///_aCentroid[0] = ; _aCentroid[1] = 1.5f; _aCentroid[2] = 0.3f; 
			_bVBOsPBOsCreated = false;

			_bRenderNormal = false;
			_bEnableLighting = false;
			_fSize = 0.2f;
	}

	void CGLUtil::clearColorDepth()
	{
		glClearColor ( 1.f,1.f,1.f,1.0f );
		//glClearColor ( 0.1f,0.1f,0.4f,1.0f );
		glClearDepth ( 1.0 );
	}
	
	void CGLUtil::constructVBOsPBOs(){
		if( _bVBOsPBOsCreated ) return;

		int nRGBBaseH = _uResolution != 6? btl::kinect::__aRGBH[0] : btl::kinect::__aRGBH[_uResolution];
		int nRGBBaseW = _uResolution != 6? btl::kinect::__aRGBW[0] : btl::kinect::__aRGBW[_uResolution];

		int nDepthBaseH = btl::kinect::__aDepthH[0];
		int nDepthBaseW = btl::kinect::__aDepthW[0];

		for (ushort usLevel=0; usLevel<4; usLevel++){
			int nRowsRGB = nRGBBaseH >> usLevel;
			int nColsRGB = nRGBBaseW >> usLevel;//__aKinectW[_uResolution]>>i;

			int nRowsDepth = nDepthBaseH >> usLevel;
			int nColsDepth = nDepthBaseW >> usLevel;

			createVBO( nColsDepth, nRowsDepth,3,sizeof(float),&_auPtVBO[usLevel], &_apResourcePtVBO[usLevel] );
			createVBO( nColsDepth, nRowsDepth,3,sizeof(float),&_auNlVBO[usLevel], &_apResourceNlVBO[usLevel] );
			createVBO( nColsDepth, nRowsDepth,3,sizeof(uchar),&_auRGBVBO[usLevel],&_apResourceRGBVBO[usLevel]); //this is used for color each vertex in 3d
			createPBO( nColsRGB, nRowsRGB, 3, sizeof(uchar), &_auRGBPixelBO[usLevel],&_apResourceRGBPxielBO[usLevel],&_auTexture[usLevel]);//this is used for loading textures
			createPBO( nColsRGB, nRowsRGB, 1, sizeof(uchar), &_auGrayPixelBO[usLevel], &_apResourceGrayPxielBO[usLevel], &_auGrayTexture[usLevel]);//this is used for loading textures
		}//for each pyramid level
		_bVBOsPBOsCreated = true;
	}//constructVBOsPBOs()
	void CGLUtil::createVBO(const unsigned int uRows_, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_,
		GLuint* puVBO_, cudaGraphicsResource** ppResourceVBO_ ){
			// the first four are standard OpenGL, the 5th is the CUDA reg 
			// of the VBO these calls exist starting in OpenGL 1.5
			glGenBuffers(1, puVBO_);
			glBindBuffer(GL_ARRAY_BUFFER, *puVBO_);
			glBufferData(GL_ARRAY_BUFFER, uRows_*uCols_*usChannel_*usBytes_, 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			cudaSafeCall( cudaGraphicsGLRegisterBuffer( ppResourceVBO_, *puVBO_, cudaGraphicsMapFlagsWriteDiscard) );
	}//createVBO()

	void CGLUtil::createPBO(const unsigned int uRows_, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_, GLuint* puPBO_, cudaGraphicsResource** ppResourcePixelBO_, GLuint* pTexture_){
		//Generate a buffer ID called a PBO (Pixel Buffer Object)
		//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
		//generate a texture
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, pTexture_);
		glBindTexture ( GL_TEXTURE_2D, *pTexture_ );
		glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
		glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
		glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
		glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture  
		// 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
		// border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, uCols_, uRows_, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL ); //???????????????????
		glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_MIN_FILTER , GL_NEAREST);
		glBindTexture( GL_TEXTURE_2D, 0);
		//generate PBO
		glGenBuffers(1, puPBO_);
		//Make this the current UNPACK buffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *puPBO_);
		//Allocate data for the buffer. 4-channel 8-bit image
		glBufferData(GL_PIXEL_UNPACK_BUFFER, uRows_*uCols_*	usChannel_ *usBytes_, NULL, GL_STREAM_DRAW); //GL_STREAM_DRAW //http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0 );
		cudaSafeCall( cudaGraphicsGLRegisterBuffer( ppResourcePixelBO_, *puPBO_, cudaGraphicsRegisterFlagsNone) );//cudaGraphicsRegisterFlagsWriteDiscard) ); //
		//cudaSafeCall( cudaGLRegisterBufferObject(*puPBO_) ); //deprecated
	}//createVBO()

	void CGLUtil::destroyVBOsPBOs(){
		if( !_bVBOsPBOsCreated ) return;
		for (ushort usLevel=0; usLevel<_usPyrHeight; usLevel++){
			//destroyVBOsPBOs(u);
			releaseVBO( _auPtVBO[usLevel], _apResourcePtVBO[usLevel] );
			releaseVBO( _auNlVBO[usLevel], _apResourceNlVBO[usLevel] );
			releaseVBO( _auRGBVBO[usLevel],_apResourceRGBVBO[usLevel]);
			releasePBO( _auRGBPixelBO[usLevel],_apResourceRGBPxielBO[usLevel]);
			releasePBO( _auGrayPixelBO[usLevel], _apResourceGrayPxielBO[usLevel]);
		}//for each pyramid level
		_bVBOsPBOsCreated = false;
	}

	void CGLUtil::drawString(const char *str, int x, int y, float color[4], void *font) const
	{
		// backup current model-view matrix
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();                     // save current modelview matrix
		glLoadIdentity();                   // reset modelview matrix

		// set to 2D orthogonal projection
		glMatrixMode(GL_PROJECTION);        // switch to projection matrix
		glPushMatrix();                     // save current projection matrix
		glLoadIdentity();                   // reset projection matrix
		gluOrtho2D(0, 1280, 0, 480);  // set to orthogonal projection

		glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT); // lighting and color mask
		glDisable(GL_LIGHTING);     // need to disable lighting for proper text color
		glDisable(GL_TEXTURE_2D);

		glColor4fv(color);          // set text color
		glRasterPos2i(x, y);        // place text position

		// loop all characters in the string
		while(*str)
		{
			glutBitmapCharacter(font, *str);
			
			++str;
		}

		//glEnable(GL_TEXTURE_2D);
		//glEnable(GL_LIGHTING);
		glPopAttrib();

		// restore projection matrix
		glPopMatrix();                   // restore to previous projection matrix

		// restore modelview matrix
		glMatrixMode(GL_MODELVIEW);      // switch to modelview matrix
		glPopMatrix();                   // restore to previous modelview matrix
	}

	void CGLUtil::errorDetectorGL() const
	{
		GLenum eError = glGetError();
		if (eError != GL_NO_ERROR)
		{
			switch(eError){
			case GL_INVALID_ENUM:
				PRINTSTR("GL_INVALID_ENUM");break;
			case GL_INVALID_VALUE:
				PRINTSTR("GL_INVALID_VALUE");break;
			case GL_INVALID_OPERATION:
				PRINTSTR("GL_INVALID_OPERATION");break;
			case GL_STACK_OVERFLOW:
				PRINTSTR("GL_STACK_OVERFLOW");break;
			case GL_STACK_UNDERFLOW:
				PRINTSTR("GL_STACK_UNDERFLOW");break;
			case GL_OUT_OF_MEMORY:
				PRINTSTR("GL_OUT_OF_MEMORY");break;
			}
		}
	}
	void CGLUtil::getRTFromWorld2CamCV(Eigen::Matrix3f* pRw_, Eigen::Vector3f* pTw_) {
		//only the matrix in the top of the modelview matrix stack works
		Eigen::Affine3f M;
		glGetFloatv(GL_MODELVIEW_MATRIX,M.matrix().data());

		Eigen::Matrix3f S;
		*pTw_ = M.translation();
		*pRw_ = M.linear();
		//M.computeRotationScaling(pRw_,&S);
		//*pTw_ = (*pTw_)/S(0,0);
		//negate row no. 1 and 2, to switch from GL to CV convention
		for (int r = 1; r < 3; r++){
			for	(int c = 0; c < 3; c++){
				(*pRw_)(r,c) = -(*pRw_)(r,c);
			}
			(*pTw_)(r) = -(*pTw_)(r); 
		}
		//PRINT(S);
		//PRINT(*pRw_);
		//PRINT(*pTw_);
		return;
	}
	int CGLUtil::getLevel(int nCols_ ){
		switch ( nCols_ )
		{
		case 1280:
			return 0;
		case 640:
			return 0;
		case 320:
			return 1;
		case 160:
			return 2;
		case 80:
			return 3;
		case 40:
			return 4;
		case 20:
			return 5;
		default:
			cout << nCols_ << endl;
			PRINTSTR("Failure - input resolution is not 640x480 or 320x240 or 160x120 or 80x60 or 40x30. ");
			return -1;
		}
		return -1;
	}
	void CGLUtil::gpuMapPtResources(const cv::cuda::GpuMat& cvgmPts_){
		int nPyrLevel_ = getLevel( cvgmPts_.cols );
		// map OpenGL buffer object for writing from CUDA
		void *pDev;
		cudaGraphicsMapResources(1, &_apResourcePtVBO[nPyrLevel_], 0);
		size_t nSize; 
		cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourcePtVBO[nPyrLevel_] );
		cv::cuda::GpuMat cvgmPts( cvgmPts_.size(),CV_32FC3,pDev );
		cvgmPts_.copyTo(cvgmPts); // the operation of the Buffer must be done before cudaGraphicsUnmapResources(), otherwise, the buffer will affects each other
		cudaGraphicsUnmapResources(1, &_apResourcePtVBO[nPyrLevel_], 0);
		// render from the vbo
		glBindBuffer(GL_ARRAY_BUFFER, _auPtVBO[nPyrLevel_]);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);//you cant move glEnableClientState in front of cuda GraphicsMapResources, otherwise, you will have weird problem
		//glColor3f(1.0, 0.0, 0.0);
		//glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
		//glDisableClientState(GL_VERTEX_ARRAY);
		//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	void CGLUtil::gpuMapNlResources(const cv::cuda::GpuMat& cvgmNls_){
		int nPyrLevel_ = getLevel( cvgmNls_.cols );
		// map OpenGL buffer object for writing from CUDA
		void *pDev;
		cudaGraphicsMapResources(1, &_apResourceNlVBO[nPyrLevel_], 0);
		size_t nSize; 
		cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourceNlVBO[nPyrLevel_] );
		cv::cuda::GpuMat cvgmNls(cvgmNls_.size(),CV_32FC3,pDev);
		cvgmNls_.copyTo(cvgmNls);
		cudaGraphicsUnmapResources(1, &_apResourceNlVBO[nPyrLevel_], 0);
		// render from the vbo
		glBindBuffer(GL_ARRAY_BUFFER, _auNlVBO[nPyrLevel_]);
		glNormalPointer(GL_FLOAT, 12, 0); //12 is the stride = the number of bytes occupied by each normal
		glEnableClientState(GL_NORMAL_ARRAY);//you cant move glEnableClientState infront of cuda GraphicsMapResources, otherwise, you will have weird problem
		//glColor3f(1.0, 0.0, 0.0);
		//glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
		//glDisableClientState(GL_NORMAL_ARRAY);
		//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	void CGLUtil::gpuMapRGBResources(const cv::cuda::GpuMat& cvgmRGBs_){
		int nPyrLevel_ = getLevel( cvgmRGBs_.cols );
		// map OpenGL buffer object for writing from CUDA
		void *pDev;
		cudaGraphicsMapResources(1, &_apResourceRGBVBO[nPyrLevel_], 0);
		size_t nSize; 
		cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourceRGBVBO[nPyrLevel_] );
		cv::cuda::GpuMat cvgmRGBs(cvgmRGBs_.size(),CV_8UC3,pDev);
		cvgmRGBs_.copyTo(cvgmRGBs);
		cudaGraphicsUnmapResources(1, &_apResourceRGBVBO[nPyrLevel_], 0);
		// render from the vbo
		glBindBuffer(GL_ARRAY_BUFFER, _auRGBVBO[nPyrLevel_]);
		glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);//you cant move glEnableClientState infront of cuda GraphicsMapResources, otherwise, you will have weird problem
		//glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
		//glDisableClientState(GL_COLOR_ARRAY);
		//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	GLuint CGLUtil::gpuMapRgb2PixelBufferObj(const cv::cuda::GpuMat& cvgmRGB_ ){
		//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
		int nPyrLevel_ = getLevel( cvgmRGB_.cols );
		GLuint uTexture;
		// map OpenGL buffer object for writing from CUDA
		if (cvgmRGB_.channels() == 3) {
			uTexture = _auTexture[nPyrLevel_];
			void *pDev;
			cudaSafeCall( cudaGraphicsMapResources(1, &_apResourceRGBPxielBO[nPyrLevel_], 0)); 
			size_t nSize; 
			cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize , _apResourceRGBPxielBO[nPyrLevel_]));
			cv::cuda::GpuMat cvgmRGBA( cvgmRGB_.size(), CV_8UC3, pDev); 
			cvgmRGB_.copyTo(cvgmRGBA); 
			cudaSafeCall( cudaGraphicsUnmapResources(1, &_apResourceRGBPxielBO[nPyrLevel_], 0) );
			//texture mapping
			glBindTexture( GL_TEXTURE_2D, uTexture);
			glBindBuffer ( GL_PIXEL_UNPACK_BUFFER_ARB, _auRGBPixelBO[nPyrLevel_]);
			glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, cvgmRGB_.cols, cvgmRGB_.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
			errorDetectorGL();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		else if (cvgmRGB_.channels()==1) {
			uTexture = _auGrayTexture[nPyrLevel_];
			void *pDev;
			cudaSafeCall( cudaGraphicsMapResources(1, &_apResourceGrayPxielBO[nPyrLevel_], 0)); 
			size_t nSize; 
			cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize , _apResourceGrayPxielBO[nPyrLevel_]));
			cv::cuda::GpuMat cvgmRGBA( cvgmRGB_.size(), CV_8UC1, pDev);
			cvgmRGB_.copyTo(cvgmRGBA); 
			cudaSafeCall( cudaGraphicsUnmapResources(1, &_apResourceGrayPxielBO[nPyrLevel_], 0) );
			//texture mapping
			glBindTexture(GL_TEXTURE_2D, uTexture);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, _auGrayPixelBO[nPyrLevel_]);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, cvgmRGB_.cols, cvgmRGB_.rows, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
			errorDetectorGL();
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		return uTexture;
	}//gpuMapRgb2PixelBufferObj

	void CGLUtil::init()
	{
		_eimModelViewGL.setIdentity();
		//_eimModelViewGL(1, 1) = -1.f;
		//_eimModelViewGL(2, 2) = -1.f;
		//disk list
		_uDisk = glGenLists(1);
		_pQObj = gluNewQuadric();
		gluQuadricDrawStyle(_pQObj, GLU_FILL); //LINE); /* wireframe */
		gluQuadricNormals(_pQObj, GLU_SMOOTH);// FLAT);//
		glNewList(_uDisk, GL_COMPILE);
		gluDisk(_pQObj, 0.0, 0.01, 4, 1);//render a disk on z=0 plane
		glEndList();
		//normal list
		_uNormal = glGenLists(2);
		glNewList(_uNormal, GL_COMPILE);
		glDisable(GL_LIGHTING);
		glBegin(GL_LINES);
		glColor3d(1.,1.,1.);//
		glVertex3d(0.,0.,0.);
		glVertex3d(0.,0.,0.016);
		glEnd();
		glEndList();
		//voxel list
		_uVoxel = glGenLists(3);
		glNewList(_uVoxel, GL_COMPILE);
		glDisable(GL_LIGHTING);
		renderVoxelGL(1.f);
		glEndList();

		_uOctTree = glGenLists(3);
		glNewList(_uOctTree, GL_COMPILE);
		glDisable(GL_LIGHTING);
		renderOctTree<float>(0,0,0,2,1);
		glEndList();
		glEnable(GL_RESCALE_NORMAL);

		//init the global light
		//initLights();

	}//init();

	void CGLUtil::initLights()
	{
		// set up light colors (ambient, diffuse, specular)
		GLfloat lightKa[] = {.2f, .2f, .2f, 1.0f};  // ambient light
		GLfloat lightKd[] = {.7f, .7f, .7f, 1.0f};  // diffuse light
		GLfloat lightKs[] = {1, 1, 1, 1};           // specular light

		glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
		glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

		// position the light
		_aLight[0] = 2.0f;
		_aLight[1] = 1.7f;
		_aLight[2] =-0.2f;
		_aLight[3] = 1.0f;
		glLightfv(GL_LIGHT0, GL_POSITION, _aLight);

		glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration
	}



	void CGLUtil::mouseMotion ( int nX_, int nY_ )
	{
		if ( _bLButtonDown == true )
		{
			glDisable     ( GL_BLEND );
			_nXMotion = nX_ - _nXLeftDown;
			_nYMotion = nY_ - _nYLeftDown;
			_dXAngle  = _dXLastAngle + _nXMotion;
			_dYAngle  = _dYLastAngle + _nYMotion;
		}
		else if ( _bRButtonDown == true )
		{
			glDisable     ( GL_BLEND );
			_nXMotion = nX_ - _nXRightDown;
			_nYMotion = nY_ - _nYRightDown;

			_dX  = _dXLast + _nXMotion;
			_dY  = _dYLast + _nYMotion;
			_dZoom = _dZoomLast + (_nXMotion + _nYMotion)/200.;
			//PRINT(_dZoom);
		}

		glutPostRedisplay();
	}


	void CGLUtil::mouseClick ( int nButton_, int nState_, int nX_, int nY_ )
	{
		if ( nButton_ == GLUT_LEFT_BUTTON )
		{
			if ( nState_ == GLUT_DOWN )
			{
				_nXMotion = _nYMotion = 0;
				_nXLeftDown    = nX_;
				_nYLeftDown    = nY_;

				_bLButtonDown = true;
			}
			else if( nState_ == GLUT_UP )// button up
			{
				_dXLastAngle = _dXAngle;
				_dYLastAngle = _dYAngle;
				_bLButtonDown = false;
			}
			glutPostRedisplay();
		}
		else if ( GLUT_RIGHT_BUTTON )
		{
			if ( nState_ == GLUT_DOWN )
			{
				_nXMotion = _nYMotion = 0;
				_nXRightDown  = nX_;
				_nYRightDown  = nY_;
				_dZoomLast    = _dZoom;
				_bRButtonDown = true;
			}
			else if( nState_ == GLUT_UP )
			{
				_dXLast = _dX;
				_dYLast = _dY;
				_bRButtonDown = false;
			}
			glutPostRedisplay();
		}

		return;
	}
	void CGLUtil::normalKeys ( unsigned char key, int x, int y )
	{
		switch( key )
		{
		case 27:
			exit ( 0 );
			break;
		case 'g':
			//zoom in
			glDisable( GL_BLEND );
			_dZoom += _dScale;
			glutPostRedisplay();
			//PRINT( _dZoom );
			break;
		case 'h':
			//zoom out
			glDisable( GL_BLEND );
			_dZoom -= _dScale;
			glutPostRedisplay();
			//PRINT( _dZoom );
			break;
		case 'l':
			_bEnableLighting = !_bEnableLighting;
			glutPostRedisplay();
			//PRINT( _bEnableLighting );
			break;
		case 'n':
			_bRenderNormal = !_bRenderNormal;
			glutPostRedisplay();
			//PRINT( _bRenderNormal );
			break;
		case 'k':
			_fSize += 0.05f;// range from 0.05 to 1 by step 0.05
			_fSize = _fSize < 1 ? _fSize: 1;
			glutPostRedisplay();
			//PRINT( _fSize );
			break;
		case 'j':
			_fSize -= 0.05f;
			_fSize = _fSize > 0.05f? _fSize : 0.05f;
			glutPostRedisplay();
			//PRINT( _fSize );
			break;
		case '<':
			_dYAngle += 1.0;
			glutPostRedisplay();
			break;
		case '>':
			_dYAngle -= 1.0;
			glutPostRedisplay();
			break;
		case '9':
			_usLevel = ++_usLevel%_usPyrHeight;
			glutPostRedisplay();
			//PRINT(_usPyrHeight);
			break;
		case '0'://reset camera location
			setInitialPos();
			glutPostRedisplay();
			break;
		}
		return;
	}


	void CGLUtil::releasePBO( GLuint uPBO_,cudaGraphicsResource *pResourcePixelBO_ ){
		// unregister this buffer object with CUDA
		//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
		cudaSafeCall( cudaGraphicsUnregisterResource( pResourcePixelBO_ ) );
		glDeleteBuffers(1, &uPBO_);
	}//releasePBO()

	void CGLUtil::releaseVBO( GLuint uVBO_, cudaGraphicsResource *pResourceVBO_ ){
		// clean up OpenGL and CUDA
		cudaSafeCall( cudaGraphicsUnregisterResource( pResourceVBO_ ) );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		glDeleteBuffers( 1, &uVBO_ );
	}//releaseVBO()
	void CGLUtil::renderPatternGL(const float fSize_, const unsigned short usRows_, const unsigned short usCols_ ) const
	{
		GLboolean bLightIsOn;
		glGetBooleanv(GL_LIGHTING,&bLightIsOn);
		if (bLightIsOn){
			glDisable(GL_LIGHTING);
		}

		const float usStartZ = -usRows_/2*fSize_;
		const float usEndZ =    usRows_/2*fSize_;
		const float usStartX = -usCols_/2*fSize_;
		const float usEndX   =  usCols_/2*fSize_;
		glLineWidth(.01f);
		glPushMatrix();
		glColor3f ( .4f , .4f , .4f );
		glBegin ( GL_LINES );
		//render rows
		for ( unsigned short r = 0; r <= usRows_; r++ ){
			glVertex3f ( usStartX,  0, usStartZ+r*fSize_ );
			glVertex3f ( usEndX,    0, usStartZ+r*fSize_ );
		}
		//render cols
		for ( unsigned short c = 0; c <= usCols_; c++ ){
			glVertex3f ( usStartX+c*fSize_,  0, usStartZ );
			glVertex3f ( usStartX+c*fSize_,  0, usEndZ );
		}
		glEnd();
		glPopMatrix();

		if (bLightIsOn){
			glEnable(GL_LIGHTING);
		}
		return;
	}

	void CGLUtil::renderAxisGL() const
	{
		glDisable(GL_LIGHTING);

		glPushMatrix();
		float fAxisLength = 1.f;
		float fLengthWidth = 1;
		Eigen::Vector3f vOrigin,vXAxis,vYAxis,vZAxis;
		vOrigin<< .0f, .0f, .0f;
		vXAxis << fAxisLength, .0f, .0f;
		vYAxis << .0f, fAxisLength, .0f;
		vZAxis << .0f, .0f, fAxisLength;

		glLineWidth( fLengthWidth );
		// x axis
		glColor3f ( 1.f, .0f, .0f );
		glBegin ( GL_LINES );
		glVertex3fv ( vOrigin.data() );
		glVertex3fv ( vXAxis.data() );
		glEnd();
		// y axis
		glColor3f ( .0f, 1.f, .0f );
		glBegin ( GL_LINES );
		glVertex3fv ( vOrigin.data() );
		glVertex3fv ( vYAxis.data() );
		glEnd();
		// z axis
		glColor3f ( .0f, .0f, 1.f );
		glBegin ( GL_LINES );
		glVertex3fv ( vOrigin.data() );
		glVertex3fv ( vZAxis.data());
		glEnd();
		glPopMatrix();
	}

	void CGLUtil::renderTestPlane(){
		glPushMatrix();
		glTranslatef(1.5f,1.5f,0.3f);
		glScalef(0.15f,0.15f,0.15f);
		glColor3f(1.f,0.f,0.f);
		glBegin(GL_QUADS);
		glVertex3f(-1.f, -1.f, 0.f);
		glVertex3f(-1.f, 1.f, 0.f);
		glVertex3f(1.f, 1.f, 0.f);
		glVertex3f(1.f, -1.f, 0.f);
		glEnd();
		glPopMatrix();
	}
	void CGLUtil::renderVoxelGL2( const float fSize_) const{
		float fHS = fSize_/2.f;
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( fHS, fHS, fHS ); 
		glVertex3f ( fHS, fHS,-fHS ); 
		glVertex3f (-fHS, fHS,-fHS ); 
		glVertex3f (-fHS, fHS, fHS ); 
		glEnd();
		//bottom
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( fHS,-fHS, fHS ); 
		glVertex3f ( fHS,-fHS,-fHS ); 
		glVertex3f (-fHS,-fHS,-fHS ); 
		glVertex3f (-fHS,-fHS, fHS ); 
		glEnd();
		//middle
		glBegin ( GL_LINES );
		glVertex3f ( fHS, fHS, fHS ); 
		glVertex3f ( fHS,-fHS, fHS ); 
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f ( fHS, fHS,-fHS ); 
		glVertex3f ( fHS,-fHS,-fHS ); 
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f (-fHS, fHS,-fHS ); 
		glVertex3f (-fHS,-fHS,-fHS ); 
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f (-fHS, fHS, fHS ); 
		glVertex3f (-fHS,-fHS, fHS ); 
		glEnd();
	}
	void CGLUtil::renderVoxelGL( const float fSize_) const
	{
		// x axis
		glColor3f ( 1.f, .0f, .0f );
		//top
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( 0.f,    0.f, 0.f ); 
		glVertex3f ( fSize_, 0.f, 0.f ); 
		glVertex3f ( fSize_, 0.f, fSize_ ); 
		glVertex3f ( 0.f,    0.f, fSize_ ); 
		glEnd();
		//bottom
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( 0.f,    fSize_, 0.f ); 
		glVertex3f ( fSize_, fSize_, 0.f ); 
		glVertex3f ( fSize_, fSize_, fSize_ ); 
		glVertex3f ( 0.f,    fSize_, fSize_ ); 
		glEnd();
		//middle
		glBegin ( GL_LINES );
		glVertex3f ( 0.f,    0.f, 0.f ); 
		glVertex3f ( 0.f,    fSize_, 0.f );
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f ( fSize_, 0.f, 0.f ); 
		glVertex3f ( fSize_, fSize_, 0.f );
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f ( fSize_, 0.f,   fSize_ ); 
		glVertex3f ( fSize_, fSize_,fSize_ );
		glEnd();
		glBegin ( GL_LINES );
		glVertex3f ( 0.f, 0.f,    fSize_ ); 
		glVertex3f ( 0.f, fSize_, fSize_ );
		glEnd();
	}

	void CGLUtil::setCudaDeviceForGLInteroperation() {
		cudaDeviceProp  sProp;
		memset( &sProp, 0, sizeof( cudaDeviceProp ) );
		sProp.major = 1;
		sProp.minor = 0;
		int nDev;
		cudaSafeCall( cudaChooseDevice( &nDev, &sProp ) );
		// tell CUDA which nDev we will be using for graphic interop
		// from the programming guide:  Interoperability with OpenGL
		//     requires that the CUDA nDeviceNO_ be specified by
		//     cudaGLSetGLDevice() before any other runtime calls.
		//cudaSafeCall( cudaGLSetGLDevice( nDev ) ;

		return;
	}//setCudaDeviceForGLInteroperation()

	void CGLUtil::setInitialPos(){
		
		_eimModelViewGL.setIdentity();
		//_eimModelViewGL(1, 1) = -1.f;
		//_eimModelViewGL(2, 2) = -1.f;

		_dXAngle = 0.;
		_dYAngle = 0.;
		_dZoom = 0.;
	}

	void CGLUtil::setOrthogonal( )
	{
		// set orthographic viewing frustum
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0,btl::kinect::__aDepthW[0], 0, btl::kinect::__aDepthH[0], -1, 1);

		return;
	}

	void CGLUtil::specialKeys( int key, int x, int y ){
		switch ( key ) {
		case GLUT_KEY_F2: //display camera
			_bDisplayCamera = !_bDisplayCamera;
			glutPostRedisplay();
			break;
		case GLUT_KEY_F3:
			_bRenderReference = !_bRenderReference;
			glutPostRedisplay();
			break;
		}
	}

	void CGLUtil::viewerGL()
	{
		glMatrixMode(GL_MODELVIEW);
		// load the matrix to set camera pose
		glLoadMatrixf(_eimModelViewGL.data());
		
		//rotation
		Eigen::Matrix3f eimRotation;
		if( btl::utility::BTL_GL == _eConvention ){
			eimRotation = Eigen::AngleAxisf(float(_dXAngle*M_PI/180.f), Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(float(_dYAngle*M_PI/180.f), Eigen::Vector3f::UnitX());                         // 3. rotate horizontally
		}//mouse x-movement is the rotation around y-axis
		else if( btl::utility::BTL_CV == _eConvention )	{
			eimRotation = Eigen::AngleAxisf(float(_dXAngle*M_PI/180.f), -Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(float(_dYAngle*M_PI/180.f), Eigen::Vector3f::UnitX());                         // 3. rotate horizontally
		}
		//translation
		/*_dZoom = _dZoom < 0.1? 0.1: _dZoom;
		_dZoom = _dZoom > 10? 10: _dZoom;*/

		//get direction N pointing from camera center to the object centroid
		Eigen::Affine3f M; M = _eimModelViewGL;
		Eigen::Vector3f T = M.translation();
		Eigen::Matrix3f R = M.linear();
		Eigen::Vector3f C = -R.transpose()*T;//camera centre
		Eigen::Vector3f N = _eivCentroid - C;//from camera centre to object centroid
		N = N/N.norm();//normalization

		Eigen::Affine3f _eimManipulate; _eimManipulate.setIdentity();
		_eimManipulate.translate( N*float(_dZoom) );//(N*(1-_dZoom));  //use camera movement toward object for zoom in/out effects
		_eimManipulate.translate(_eivCentroid);  // 5. translate back to the original camera pose
		//_eimManipulate.scale(s);				 // 4. zoom in/out, never use scale to simulate camera movement, it affects z-buffer capturing. use translation instead
		_eimManipulate.rotate(eimRotation);		 // 2. rotate vertically // 3. rotate horizontally
		_eimManipulate.translate(-_eivCentroid); // 1. translate the camera center to align with object centroid*/
		glMultMatrixf(_eimManipulate.data());

		/*	
		lTranslated( _aCentroid[0], _aCentroid[1], _aCentroid[2] ); // 5. translate back to the original camera pose
		_dZoom = _dZoom < 0.1? 0.1: _dZoom;
		_dZoom = _dZoom > 10? 10: _dZoom;
		glScaled( _dZoom, _dZoom, _dZoom );                      //  4. zoom in/out, 
		if( btl::utility::BTL_GL == _eConvention )
		glRotated ( _dXAngle, 0, 1 ,0 );                         // 3. rotate horizontally
		else if( btl::utility::BTL_CV == _eConvention )			//mouse x-movement is the rotation around y-axis
		glRotated ( _dXAngle, 0,-1 ,0 ); 
		glRotated ( _dYAngle, 1, 0 ,0 );                             // 2. rotate vertically
		glTranslated(-_aCentroid[0],-_aCentroid[1],-_aCentroid[2] ); // 1. translate the camera center to align with object centroid
		*/

		// light position in 3d
		glLightfv(GL_LIGHT0, GL_POSITION, _aLight);
	}

	void CGLUtil::initCuda() {
		cudaSafeCall( cudaSetDevice( 0 ) );
		CGLUtil::printShortCudaDeviceInfo(0);
	}

	int CGLUtil::getCudaEnabledDeviceCount() 
	{
		int count;
		cudaError_t error = cudaGetDeviceCount( &count );

		if (error == cudaErrorInsufficientDriver)
			return -1;

		if (error == cudaErrorNoDevice)
			return 0;

		cudaSafeCall(error);
		return count;  
	}

	void CGLUtil::printShortCudaDeviceInfo(int nDeviceNO_) 
	{
		int nDeviceCount = getCudaEnabledDeviceCount();
		bool valid = (nDeviceNO_ >= 0) && (nDeviceNO_ < nDeviceCount);

		int beg = valid ? nDeviceNO_   : 0;
		int end = valid ? nDeviceNO_+1 : nDeviceCount;

		int driverVersion = 0, runtimeVersion = 0;
		cudaSafeCall( cudaDriverGetVersion(&driverVersion) );
		cudaSafeCall( cudaRuntimeGetVersion(&runtimeVersion) );

		for(int dev = beg; dev < end; ++dev)
		{                
			cudaDeviceProp prop;
			cudaSafeCall( cudaGetDeviceProperties(&prop, dev) );

			const char *arch_str = prop.major < 2 ? " (pre-Fermi)" : "";
			printf("Device %d:  \"%s\"  %.0fMb", dev, prop.name, (float)prop.totalGlobalMem/1048576.0f);                
			printf(", sm_%d%d%s, %d cores", prop.major, prop.minor, arch_str, /*convertSMVer2Cores(prop.major, prop.minor) **/ prop.multiProcessorCount);                
			printf(", Driver/Runtime ver.%d.%d/%d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);
		}
		fflush(stdout);
	}

}//gl_util
}//btl
