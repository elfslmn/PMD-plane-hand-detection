/****************************************************************************\
 * Copyright (C) 2017 Infineon Technologies & pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>

#include <DetectionParams.h>
#include "stdafx.h"
#include "PlaneDetector.h"
#include "Util.h"
#include "HandDetector.h"

using namespace royale;
using namespace std;
using namespace cv;
using namespace ark;

// Linker errors for the OpenCV sample
//
// If this example gives linker errors about undefined references to cv::namedWindow and cv::imshow,
// or QFontEngine::glyphCache and qMessageFormatString (from OpenCV to Qt), it may be caused by a
// change in the compiler's C++ ABI.
//
// With Ubuntu and Debian's distribution packages, the libopencv packages that have 'v5' at the end
// of their name, for example libopencv-video2.4v5, are compatible with GCC 5 (and GCC 6), but
// incompatible with GCC 4.8 and GCC 4.9. The -dev packages don't have the postfix, but depend on
// the v5 (or non-v5) version of the corresponding lib package.  When Ubuntu moves to OpenCV 3.0,
// they're likely to drop the postfix (but the packages will be for GCC 5 or later).
//
// If you are manually installing OpenCV or Qt, you need to ensure that the binaries were compiled
// with the same version of the compiler.  The version number of the packages themselves doesn't say
// which ABI they use, it depends on which version of the compiler was used.

class CamListener : public IDepthDataListener
{

public :

    CamListener() :
        undistortImage (false)
    {
    }

    void onNewData (const DepthData *data)
    {
        // this callback function will be called for every new depth frame

        std::lock_guard<std::mutex> lock (flagMutex);

        // create two images which will be filled afterwards
        // each image containing one 32Bit channel
        zImage.create (Size (data->width, data->height), CV_32FC1);
        grayImage.create (Size (data->width, data->height), CV_32FC1);
        xyz_map.create(Size(data->width, data->height), CV_32FC3);
        confidence_map.create(Size(data->width, data->height), CV_8UC1);

        // set the image to zero
        zImage = Scalar::all (0);
        grayImage = Scalar::all (0);
        xyz_map = Scalar::all(0);

        int k = 0;
        for (int y = 0; y < xyz_map.rows; y++)
        {
            Vec3f *xyzptr = xyz_map.ptr<Vec3f>(y);
            float *grayptr = grayImage.ptr<float>(y);
            uint8_t *confptr = confidence_map.ptr<uint8_t>(y);
            for (int x = 0; x < xyz_map.cols; x++, k++)
            {
                auto curPoint = data->points.at (k);
                confptr[x] = curPoint.depthConfidence;
                if(curPoint.z == 0.0f)
                {
                   continue; //invalid depth
                }
                if (curPoint.z < NOISE_FILTER_LOW || curPoint.z  > NOISE_FILTER_HIGH || curPoint.depthConfidence< 60)
                {
                   continue;
                }
                xyzptr[x][0] = curPoint.x;
                xyzptr[x][1] = curPoint.y;
                xyzptr[x][2] = curPoint.z;
                grayptr[x] = curPoint.grayValue;
            }
        }
        planeDetector->update(xyz_map);
        handDetector->update(xyz_map);
        hands = handDetector->getHands();
        //cout <<"Hand found: "<<hands.size() << endl;
        //planeDetector->detectRansac(xyz_map);

        // vector<Mat> channels(3);
        // split(xyz_map, channels);
        // zImage = channels[2]; // debug
        //
        // // create images to store the 8Bit version (some OpenCV
        // // functions may only work on 8Bit images)
        // zImage8.create (Size (data->width, data->height), CV_8UC1);
        // grayImage8.create (Size (data->width, data->height), CV_8UC1);
        //
        // // this normalizes the images from min/max to 0/255
        // normalize (zImage, zImage8, 0, 255, NORM_MINMAX, CV_8UC1);
        // normalize (grayImage, grayImage8, 0, 255, NORM_MINMAX, CV_8UC1);
        //
        // if (undistortImage)
        // {
        //     // call the undistortion function on the z image
        //     Mat temp = zImage8.clone();
        //     undistort (temp, zImage8, cameraMatrix, distortionCoefficients);
        // }
        // imshow ("Depth", zImage8);
        //
        // if (undistortImage)
        // {
        //     // call the undistortion function on the gray image
        //     Mat temp = grayImage8.clone();
        //     undistort (temp, grayImage8, cameraMatrix, distortionCoefficients);
        // }
        // imshow ("Gray", grayImage8);
    }

    void setLensParameters (LensParameters lensParameters)
    {
        // Construct the camera matrix
        // (fx   0    cx)
        // (0    fy   cy)
        // (0    0    1 )
        cameraMatrix = (Mat1d (3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
                        0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
                        0, 0, 1);

        // Construct the distortion coefficients
        // k1 k2 p1 p2 k3
        distortionCoefficients = (Mat1d (1, 5) << lensParameters.distortionRadial[0],
                                  lensParameters.distortionRadial[1],
                                  lensParameters.distortionTangential.first,
                                  lensParameters.distortionTangential.second,
                                  lensParameters.distortionRadial[2]);
    }

    void toggleUndistort()
    {
        std::lock_guard<std::mutex> lock (flagMutex);
        undistortImage = !undistortImage;
    }

    void setPlaneDetector(PlaneDetector::Ptr detector)
    {
      planeDetector = detector;
    }
    void setHandDetector(HandDetector::Ptr detector)
    {
      handDetector = detector;
    }

    bool saveFrame(std::string destination)
    {
        cv::FileStorage fs(destination, cv::FileStorage::WRITE);
        std::lock_guard<std::mutex> lock (flagMutex);

        fs << "xyz_map" << xyz_map;
        fs.release();
        return true;
    }

private:

    // define images for depth and gray
    // and for their 8Bit
    Mat zImage, zImage8;
    Mat grayImage, grayImage8;
    Mat xyz_map;
    Mat confidence_map;

    // lens matrices used for the undistortion of
    // the image
    Mat cameraMatrix;
    Mat distortionCoefficients;

    PlaneDetector::Ptr planeDetector;
    HandDetector::Ptr handDetector;
    std::vector<Hand::Ptr> hands;
   //Minimum depth of points (in meters). Points under this depth are presumed to be noise. (0.0 to disable)
   const float NOISE_FILTER_LOW = 0.14f;
   //Maximum depth of points (in meters). Points above this depth are presumed to be noise. (0.0 to disable)
   const float NOISE_FILTER_HIGH = 0.99f;

    std::mutex flagMutex;
    bool undistortImage;
};

Mat readImage(std::string source)
{
   Mat xyz_map;
    cv::FileStorage fs;
    fs.open(source, cv::FileStorage::READ);
    fs["xyz_map"] >> xyz_map;
    fs.release();

    return xyz_map;
}

int main (int argc, char *argv[])
{
   // OpenARK variables
   // initialize detection parameters
   DetectionParams::Ptr params = DetectionParams::create(); // default parameters
   // initialize detectors
   PlaneDetector::Ptr planeDetector = std::make_shared<PlaneDetector>();
   HandDetector::Ptr handDetector = std::make_shared<HandDetector>(planeDetector);
   handDetector->setParams(params);

    // This is the data listener which will receive callbacks.  It's declared
    // before the cameraDevice so that, if this function exits with a 'return'
    // statement while the camera is still capturing, it will still be in scope
    // until the cameraDevice's destructor implicitly de-registers the listener.
    CamListener listener;

    // this represents the main camera device object
    std::unique_ptr<ICameraDevice> cameraDevice;

    // the camera manager will query for a connected camera
    {
      CameraManager manager;
      // if no argument was given try to open the first connected camera
      royale::Vector<royale::String> camlist (manager.getConnectedCameraList());
      cout << "Detected " << camlist.size() << " camera(s)." << endl;

      if (!camlist.empty())
      {
          cameraDevice = manager.createCamera (camlist[0]);
      }
      else
      {
          cerr << "No suitable camera device detected." << endl
               << "Please make sure that a supported camera is plugged in, all drivers are "
               << "installed, and you have proper USB permission" << endl;
          return 1;
      }

      camlist.clear();

    }
    // the camera device is now available and CameraManager can be deallocated here
    if (cameraDevice == nullptr)
    {
      // no cameraDevice available
      cerr << "Cannot create the camera device" << endl;
      return 1;
    }

    // IMPORTANT: call the initialize method before working with the camera device
    auto status = cameraDevice->initialize();
    if (status != CameraStatus::SUCCESS)
    {
        cerr << "Cannot initialize the camera device, error string : " << getErrorString (status) << endl;
        return 1;
    }

    royale::Vector<royale::String> opModes;
    royale::Pair<uint32_t, uint32_t> exposureLimits;

    status = cameraDevice->getUseCases (opModes);
    if (status != CameraStatus::SUCCESS)
    {
      printf ("Failed to get use cases, CODE %d", (int) status);
   }

    // retrieve the lens parameters from Royale
    LensParameters lensParameters;
    status = cameraDevice->getLensParameters (lensParameters);
    if (status != CameraStatus::SUCCESS)
    {
        cerr << "Can't read out the lens parameters" << endl;
        return 1;
    }

    listener.setLensParameters (lensParameters);
    listener.setPlaneDetector(planeDetector);
    listener.setHandDetector(handDetector);

    // register a data listener
    if (cameraDevice->registerDataListener (&listener) != CameraStatus::SUCCESS)
    {
        cerr << "Error registering data listener" << endl;
        return 1;
    }

    // set an operation mode
    int opMode = 0;
    status = cameraDevice->setUseCase (opModes[opMode]);
    if (status != CameraStatus::SUCCESS)
    {
      printf ("Failed to set use case, CODE %d", (int) status);
   }else{
      printf ("Operation mode: %s\n", opModes.at (opMode).c_str());
   }

   //set exposure mode to manual
   status = cameraDevice->setExposureMode (ExposureMode::AUTOMATIC);
   if (status != CameraStatus::SUCCESS)
   {
      printf ("Failed to set exposure mode, CODE %d\n", (int) status);
   }

    // create two windows
    //namedWindow ("Depth", WINDOW_AUTOSIZE);
    //namedWindow ("Gray", WINDOW_AUTOSIZE);
    namedWindow ("FloodFill", WINDOW_AUTOSIZE);
    namedWindow ("NormalMap", WINDOW_AUTOSIZE);
    namedWindow ("[Plane Debug]", WINDOW_AUTOSIZE);
    //namedWindow ("Inliers", WINDOW_AUTOSIZE);
    namedWindow ("[Hand Flood Fill Debug]", WINDOW_AUTOSIZE);

    //start capture mode
    if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error starting the capturing" << endl;
        return 1;
    }

    // Mat xyz_map = readImage(argv[1]);
    // vector<Mat> channels(3);
    // split(xyz_map, channels);
    // planeDetector->update(xyz_map);

    //eqn = util::ransacFindPlane(points, 0.1f,atoi(argv[2]));

    int currentKey = 0;

    while (currentKey != 27)
    {
        currentKey = waitKey (0) & 255;

        if (currentKey == 'd')
        {
            // toggle the undistortion of the image
            listener.toggleUndistort();
        }
        else if (currentKey == 's')
        {
            listener.saveFrame("xyz2");
        }
        else if (currentKey == 'p')
        {
           if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
           {
               cerr << "Error stopping the capturing" << endl;
               return 1;
           }
        }
        else if (currentKey == 'o')
        {
           if (cameraDevice->startCapture() != CameraStatus::SUCCESS)
          {
              cerr << "Error starting the capturing" << endl;
              return 1;
          }
        }
    }

    //stop capture mode
    if (cameraDevice->stopCapture() != CameraStatus::SUCCESS)
    {
        cerr << "Error stopping the capturing" << endl;
        return 1;
    }

    return 0;
}
