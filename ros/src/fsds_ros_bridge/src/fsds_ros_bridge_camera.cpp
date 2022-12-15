#include "common/common_utils/StrictMode.hpp"
#include "ros/ros.h"
#include <ros/console.h>
#include <ros/spinner.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/String.h>
#include "common/CommonStructs.hpp"
#include "common/AirSimSettings.hpp"
#include "common/Common.hpp"
#include "vehicles/car/api/CarRpcLibClient.hpp"
#include "statistics.h"
#include "rpc/rpc_error.h"
#include <cv_bridge/cv_bridge.h>
#include <math.h>
#include <boost/assign/list_of.hpp>


typedef msr::airlib::ImageCaptureBase::ImageRequest ImageRequest;
typedef msr::airlib::ImageCaptureBase::ImageResponse ImageResponse;
typedef msr::airlib::ImageCaptureBase::ImageType ImageType;
typedef msr::airlib::CameraInfo CameraInfo;

// number of seconds to record frames before printing fps
const int FPS_WINDOW = 3;

msr::airlib::CarRpcLibClient* airsim_api;
ros::Publisher image_pub;
ros::Publisher image_info_pub;
ros_bridge::Statistics fps_statistic;

// settings
std::string camera_name = "";
double framerate = 0.0;
std::string host_ip = "localhost";
bool depthcamera = false;

ros::Time make_ts(uint64_t unreal_ts)
{
   // unreal timestamp is a unix nanosecond timestamp
   // Ros also uses unix timestamps, as long as it is not running in simulated time mode.
   // For now we are not supporting simulated time.
   ros::Time out;
   return out.fromNSec(unreal_ts);
}

std::string logprefix() {
    return "[cambridge " + camera_name + "] ";
}

std::vector<ImageResponse> getImage(ImageRequest req) {
     // We are using simGetImages instead of simGetImage because the latter does not return image dimention information.
    std::vector<ImageRequest> reqs;
    reqs.push_back(req);

    std::vector<ImageResponse> img_responses;
    try {
        img_responses = airsim_api->simGetImages(reqs, "FSCar");
    } catch (rpc::rpc_error& e) {
        std::cout << "error" << std::endl;
        std::string msg = e.get_error().as<std::string>();
        std::cout << "Exception raised by the API while getting image:" << std::endl
                << msg << std::endl;
    }
    return img_responses;
}

CameraInfo getCameraInfo(const std::string camera_name, const std::string vehicle_name) {
    CameraInfo response;
    try {
        response = airsim_api->simGetCameraInfo(camera_name, vehicle_name);
        return response;
    } catch (rpc::rpc_error& e) {
        std::cout << "error" << std::endl;
        std::string msg = e.get_error().as<std::string>();
        std::cout << "Exception raised by the API while getting image info:" << std::endl
                << msg << std::endl;
    }
}

void doImageUpdate(const ros::TimerEvent&)
{
    auto img_responses = getImage(ImageRequest(camera_name, ImageType::Scene, false, false));
    CameraInfo cam_info_response = getCameraInfo(camera_name, "FSCar");


    // if a render request failed for whatever reason, this img will be empty.
    if (img_responses.size() == 0 || img_responses[0].time_stamp == 0)
        return;

    ImageResponse img_response = img_responses[0];

    sensor_msgs::ImagePtr img_msg = boost::make_shared<sensor_msgs::Image>();

    img_msg->data = img_response.image_data_uint8;
    img_msg->height = img_response.height;
    img_msg->width = img_response.width;
    img_msg->step = img_response.width * 3; // image_width * num_bytes
    img_msg->encoding = "bgr8";
    img_msg->is_bigendian = 0;
    img_msg->header.stamp = make_ts(img_response.time_stamp);
    img_msg->header.frame_id = "/fsds/camera/"+camera_name;

    sensor_msgs::CameraInfo cam_info_msg;
    cam_info_msg.header.frame_id = "/fsds/camera/" + camera_name + "/camera_info";
    cam_info_msg.header.stamp = make_ts(img_response.time_stamp);
    cam_info_msg.height = img_response.height;
    cam_info_msg.width = img_response.width;
    cam_info_msg.distortion_model = "plump_bob";
    boost::array< double, 12 > flattened_proj_mat;
    
    for (auto q = 0; q < 4; q++)
    {
        for (auto t = 0; t < 3; t++)
        {
            flattened_proj_mat[q * 3 + t] = cam_info_response.proj_mat.matrix[q][t];
        }
    }
    
    cam_info_msg.P = flattened_proj_mat;
    float f_x, f_y = cam_info_msg.width / 2;
    cam_info_msg.K = {f_x, 0, cam_info_msg.width / 2, 0, f_y,  cam_info_msg.height / 2, 0, 0, 1};

    image_pub.publish(img_msg);
    image_info_pub.publish(cam_info_msg);
    fps_statistic.addCount();
}

cv::Mat manual_decode_depth(const ImageResponse& img_response)
{
    cv::Mat mat(img_response.height, img_response.width, CV_32FC1, cv::Scalar(0));
    int img_width = img_response.width;

    for (int row = 0; row < img_response.height; row++)
        for (int col = 0; col < img_width; col++)
            mat.at<float>(row, col) = img_response.image_data_float[row * img_width + col];
    return mat;
}

float roundUp(float numToRound, float multiple) 
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple) * multiple;
}

cv::Mat noisify_depthimage(cv::Mat in)
{
    cv::Mat out = in.clone();

    // Blur
    cv::Mat kernel = cv::Mat::ones( 7, 7, CV_32F ) / (float)(7 * 7);
    cv::filter2D(in, out, -1 , kernel, cv::Point( -1, -1 ), 0, cv::BORDER_DEFAULT );

    // Decrease depth resolution
    for (int row = 0; row < in.rows; row++) {
        for (int col = 0; col < in.cols; col++) {
            float roundtarget = ceil(std::min(std::max(out.at<float>(row, col), 1.0f), 10.0f));
            out.at<float>(row, col) = roundUp(out.at<float>(row, col), roundtarget);
        }
    }

    return out;
}

void doDepthImageUpdate(const ros::TimerEvent&) {
    auto img_responses = getImage(ImageRequest(camera_name, ImageType::DepthPerspective, true, false));

    // if a render request failed for whatever reason, this img will be empty.
    if (img_responses.size() == 0 || img_responses[0].time_stamp == 0)
        return;

    ImageResponse img_response = img_responses[0];

    cv::Mat depth_img = noisify_depthimage(manual_decode_depth(img_response));
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "32FC1", depth_img).toImageMsg();
    img_msg->header.stamp = make_ts(img_response.time_stamp);
    img_msg->header.frame_id = "/fsds/camera/"+camera_name;    
    
    image_pub.publish(img_msg);
    fps_statistic.addCount();
}

void printFps(const ros::TimerEvent&)
{
    std::cout << logprefix() << "Average FPS: " << (fps_statistic.getCount()/FPS_WINDOW) << std::endl;
    fps_statistic.Reset();
}

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "fsds_ros_bridge_camera");
    ros::NodeHandle nh("~");

    // load settings
    nh.param<std::string>("camera_name", camera_name, "");
    nh.param<double>("framerate", framerate, 0.0);
    nh.param<std::string>("host_ip", host_ip, "localhost");
    nh.param<bool>("depthcamera", depthcamera, false);    

    if(camera_name == "") {
        std::cout << logprefix() << "camera_name unset." << std::endl;
        return 1;
    }
    if(framerate == 0) {
        std::cout << logprefix() << "framerate unset." << std::endl;
        return 1;
    }

    // initialize fps counter
    fps_statistic = ros_bridge::Statistics("fps");

    // ready airsim connection
    msr::airlib::CarRpcLibClient client(host_ip, RpcLibPort, 5);
    airsim_api = &client;

    try {
        airsim_api->confirmConnection();
    } catch (const std::exception &e) {
        std::string msg = e.what();
        std::cout << logprefix() << "Exception raised by the API, something went wrong." << std::endl
                  << msg << std::endl;
        return 1;
    }

    // ready topic
    image_pub = nh.advertise<sensor_msgs::Image>("/fsds/camera/" + camera_name, 1);
    image_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/fsds/camera/" + camera_name + "/camera_info", 1);


    // start the loop
    ros::Timer imageTimer = nh.createTimer(ros::Duration(1/framerate), depthcamera ? doDepthImageUpdate : doImageUpdate);
    ros::Timer fpsTimer = nh.createTimer(ros::Duration(FPS_WINDOW), printFps);
    ros::spin();
    return 0;
} 
