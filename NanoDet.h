#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <cassert>
#include <cuda_provider_factory.h>
#include "utils.h"
class NanoDet
{
public:
	NanoDet(int input_shape, float confThreshold, float nmsThreshold);
	Ort::Session load_onnx();
	cv::Mat resize_image(cv::Mat& srcimg, int* newh, int* neww, int* top, int* left);
	void post_process(std::vector<cv::Mat>& outs, cv::Mat& frame, int newh, int neww, int top, int left, std::vector<int>& box);
	void detect(cv::Mat dstimg, std::vector<int> &box_max);

private:
	const int stride[3] = { 8, 16, 32 };
	const std::string classesFile = "coco.names";
	int input_shape[2];   //// height, width
	const int reg_max = 7;
	float prob_threshold;
	float iou_threshold;
	std::vector<std::string> classes;
	int num_class;


	void softmax(float* x, int length);
	void generate_proposal(std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes, const int stride_, cv::Mat out_score, cv::Mat out_box);
	const bool keep_ratio = true;
	size_t input_tensor_size;

	std::map<const char*, std::vector<int64_t>> output_dim;
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	std::vector<int64_t> input_node_dims;
	std::vector<int64_t> output_node_dims;
	Ort::Session session = load_onnx();
};
