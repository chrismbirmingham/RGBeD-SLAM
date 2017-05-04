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


#ifndef BTL_CAMERA
#define BTL_CAMERA
#include "DllExportDef.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

namespace btl{ namespace image {

struct DLL_EXPORT SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	typedef boost::shared_ptr<SCamera> tp_scoped_ptr;
	typedef SCamera* tp_ptr;
	enum tp_camera {CAMERA_RGB, CAMERA_IR};

	//constructor
	//************************************
	// Method:    SCamera
	// FullName:  btl::image::SCamera::SCamera
	// Access:    public 
	// Returns:   na
	// Qualifier: 
	// Parameter: const std::string & strCamParam_: the yml file stores the camera internal parameters
	// Parameter: ushort uResolution_: the resolution level, where 0 is the original 1 is by half, 2 is the half of haly so on
	//************************************
	SCamera(const std::string& strCamParam_,ushort uResolution_ = 0,const string& path_ = string("..\\..\\data\\") );//0 480x640
	~SCamera();
	//methods

	//rendering
	void LoadTexture ( const cv::Mat& cvmImg_, GLuint* puTesture_ );
	void setGLProjectionMatrix ( const double dNear_, const double dFar_ );
	void renderCameraInLocal ( const GpuMat& gpu_img_, btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCoordinate_, float* color_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) ;
	void renderPointOnImageLocal(const float fX_, const float fY_, const float fPhysicalFocalLength_ = 0.1f);
	void importYML(const std::string& strCamParam_);
	void generateMapXY4Undistort();
	void gpuUndistort( const GpuMat& original_img_, GpuMat* ptr_undistorted_ ) const;

	cv::Mat getcvmK(){
		cv::Mat cvmK = ( cv::Mat_<float>(3,3) << _fFx, 0.f , _u,
												 0.f,  _fFy, _v,
										 		 0.f,  0.f , 1.f );
		return cvmK;
	}
	//camera parameters
	ushort _uResolution;
	float _fFx, _fFy, _u, _v; //_dFxIR, _dFyIR IR camera focal length
	unsigned short _sWidth, _sHeight;
	cv::Mat _cvmDistCoeffs;
	//rendering
	cv::cuda::GpuMat  _cvgmMapX;
	cv::cuda::GpuMat  _cvgmMapY;
	//type
private:
	bool _bIsUndistortionOn;
	GLUquadricObj*   _quadratic;	// Storage For Our Quadratic Objects

};

}//image
}//btl
#endif
