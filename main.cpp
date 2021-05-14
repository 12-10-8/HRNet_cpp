#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <assert.h>
#include <ctime>
#include <vector>
#include <map>
#include "argparse.h"
#include <opencv2/videoio/videoio_c.h>
#include "NanoDet.h"




std::vector<float> get_3rd_point(std::vector<float>& a, std::vector<float>& b) {
	std::vector<float> direct{ a[0] - b[0],a[1] - b[1] };
	return std::vector<float>{b[0] - direct[1], b[1] + direct[0]};
}

std::vector<float> get_dir(float src_point_x,float src_point_y,float rot_rad){
	float sn = sin(rot_rad);
	float cs = cos(rot_rad);
	std::vector<float> src_result{ 0.0,0.0 };
	src_result[0] = src_point_x * cs - src_point_y * sn;
	src_result[1] = src_point_x * sn + src_point_y * cs;
	return src_result;
}

void affine_tranform(float pt_x, float pt_y, cv::Mat& t,float* x,int p,int num) {
	float new1[3] = { pt_x, pt_y, 1.0 };
	cv::Mat new_pt(3, 1, t.type(), new1);
	cv::Mat w = t * new_pt;
	x[p] = w.at<float>(0,0);
	x[p + num] = w.at<float>(0,1);
}

cv::Mat get_affine_transform(std::vector<float>& center, std::vector<float>& scale, float rot, std::vector<int>& output_size,int inv) {
	std::vector<float> scale_tmp;
	scale_tmp.push_back(scale[0] * 200);
	scale_tmp.push_back(scale[1] * 200);
	float src_w = scale_tmp[0];
	int dst_w = output_size[0];
	int dst_h = output_size[1];
	float rot_rad = rot * 3.1415926535 / 180;
	std::vector<float> src_dir = get_dir(0, -0.5 * src_w, rot_rad);
	std::vector<float> dst_dir{ 0.0, float(-0.5) * dst_w };
	std::vector<float> src1{ center[0] + src_dir[0],center[1] + src_dir[1] };
	std::vector<float> dst0{ float(dst_w * 0.5),float(dst_h * 0.5) };
	std::vector<float> dst1{ float(dst_w * 0.5) + dst_dir[0],float(dst_h * 0.5) + dst_dir[1] };
	std::vector<float> src2 = get_3rd_point(center, src1);
	std::vector<float> dst2 = get_3rd_point(dst0, dst1);
	if (inv == 0) {
		float a[6][6] = { {center[0],center[1],1,0,0,0},
						  {0,0,0,center[0],center[1],1}, 
						  {src1[0],src1[1],1,0,0,0},
						  {0,0,0,src1[0],src1[1],1},
						  {src2[0],src2[1],1,0,0,0}, 
						  {0,0,0,src2[0],src2[1],1} };
		float b[6] = { dst0[0],dst0[1],dst1[0],dst1[1],dst2[0],dst2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
	else {
		float a[6][6] = { {dst0[0],dst0[1],1,0,0,0}, 
						  {0,0,0,dst0[0],dst0[1],1},
						  {dst1[0],dst1[1],1,0,0,0},
						  {0,0,0,dst1[0],dst1[1],1},
						  {dst2[0],dst2[1],1,0,0,0},
						  {0,0,0,dst2[0],dst2[1],1} };
		float b[6] = { center[0],center[1],src1[0],src1[1],src2[0],src2[1] };
		cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
		cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
		cv::Mat result;
		solve(a_1, b_1, result, 0);
		cv::Mat dst = result.reshape(0, 2);
		return dst;
	}
}


void transform_preds(float* coords, std::vector<float>& center, std::vector<float>& scale, std::vector<int>& output_size, std::vector<int64_t>& t, float* target_coords) {
	cv::Mat tran = get_affine_transform(center, scale, 0, output_size, 1);
	for (int p = 0; p < t[1]; ++p) {
		affine_tranform(coords[p], coords[p + t[1]], tran, target_coords, p, t[1]);
	}
}

void box_to_center_scale(std::vector<int>& box, int width, int height, std::vector<float> &center, std::vector<float> &scale) {
	int box_width = box[2] - box[0];
	int box_height = box[3] - box[1];
	center[0] = box[0] + box_width * 0.5;
	center[1] = box[1] + box_height * 0.5;
	float aspect_ratio = width * 1.0 / height;
	int pixel_std = 200;
	if (box_width > aspect_ratio * box_height) {
		box_height = box_width * 1.0 / aspect_ratio;
	}
	else if (box_width < aspect_ratio * box_height) {
		box_width = box_height * aspect_ratio;
	}
	scale[0] = box_width * 1.0 / pixel_std;
	scale[1] = box_height * 1.0 / pixel_std;
	if (center[0] != -1) {
		scale[0] = scale[0] * 1.25;
		scale[1] = scale[1] * 1.25;
	}
}

/*
* 该函数暂时只实现了batch为1的情况
*/
void get_max_preds(float* heatmap, std::vector<int64_t>& t, float* preds, float* maxvals) {
	int batch_size = t[0];
	int num_joints = t[1];
	int width = t[3];
	float* pred_mask = new float[num_joints * 2];
	int* idx = new int[num_joints * 2];
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < num_joints; ++j) {
			float max = heatmap[i * num_joints * t[2] * t[3] + j * t[2] * t[3]];
			int max_id = 0;
			for (int k = 1; k < t[2] * t[3]; ++k) {
				int index = i * num_joints * t[2] * t[3] + j * t[2] * t[3] + k;
				if (heatmap[index] > max) {
					max = heatmap[index];
					max_id = k;
				}
			}
			maxvals[j] = max;
			idx[j] = max_id;
			idx[j + num_joints] = max_id;
		}
	}
	for (int i = 0; i < num_joints; ++i) {
		idx[i] = idx[i] % width;
		idx[i + num_joints] = idx[i + num_joints] / width;
		if (maxvals[i] > 0) {
			pred_mask[i] = 1.0;
			pred_mask[i + num_joints] = 1.0;
		}
		else {
			pred_mask[i] = 0.0;
			pred_mask[i + num_joints] = 0.0;
		}
		preds[i] = idx[i] * pred_mask[i];
		preds[i + num_joints] = idx[i + num_joints] * pred_mask[i + num_joints];
	}

}

void get_final_preds(float* heatmap, std::vector<int64_t>& t, std::vector<float>& center, std::vector<float> scale, float* preds) {
	float* coords = new float[t[1] * 2];
	float* maxvals = new float[t[1]];
	int heatmap_height = t[2];
	int heatmap_width = t[3];
	get_max_preds(heatmap, t, coords, maxvals);
	for (int i = 0; i < t[0]; ++i) {
		for (int j = 0; j < t[1]; ++j) {
			int px = int(coords[i * t[1] + j] + 0.5);
			int py = int(coords[i * t[1] + j + t[1]] + 0.5);
			int index = (i * t[1] + j) * t[2] * t[3];
			if (px > 1 && px < heatmap_width - 1 && py>1 && py < heatmap_height - 1) {
				float diff_x = heatmap[index + py * t[3] + px + 1] - heatmap[index + py * t[3] + px - 1];
				float diff_y = heatmap[index + (py + 1) * t[3] + px] - heatmap[index + (py - 1) * t[3] + px];
				coords[i * t[1] + j] += sign(diff_x) * 0.25;
				coords[i * t[1] + j + t[1]] += sign(diff_y) * 0.25;
			}
		}
	}
	std::vector<int> img_size{ heatmap_width,heatmap_height };
	transform_preds(coords, center, scale, img_size, t, preds);
}

int pair_line[] = {
	//0,2,
	//2,4,
	//4,6,
	6,8,
	8,10,
	6,12,
	12,14,
	14,16,

	//0,1,
	//1,3,
	//3,5,
	5,7,
	7,9,
	5,11,
	11,13,
	13,15,
};

int main(int argc,const char* argv[])
{
;	clock_t startTime_nano, endTime_nano,startTime_hrnet,endTime_hrnet;
	std::string videopath;
	cv::VideoCapture capture;
	cv::Mat frame, frame1;
	int t = 0;
	int model = 0;
	int k = 0;
	std::string write_videopath = "D:/write.mp4";
	argparse::ArgumentParser parser("example", "Argument parser example");
	parser.add_argument("-v",
						"--video",
						"Video path",
						false
			);
	parser.add_argument("-c",
						"--camera",
						"camera index",
						false
			);
	parser.add_argument("-m",
						"--model",
						"model type,0-w48_256x192,1-w48_384x288,2-w32_256x192,3-w32_128x96",
						false
			);
	parser.add_argument("-d",
						"--display",
						"point display mode, 0-左右,1-左,2-右",
						false
			);
	parser.add_argument("-w",
		"--write_video",
		"write video path",
		false
	);
	parser.enable_help();
	parser.parse(argc, argv);
	if (parser.exists("help")) {
		parser.print_help();
		return 0;
	}
	if (parser.exists("video")) {
		videopath = parser.get<std::string>("video");
		frame = capture.open(videopath);
		k = 1;
	}
	else if (parser.exists("camera")) {
		frame = capture.open(parser.get<int>("camera"));
		k = 1;
	}
	if (parser.exists("display")) {
		t = parser.get<int>("display");
	}
	if (parser.exists("model")) {
		model = parser.get<int>("model");
	}
	if (parser.exists("write_video")) {
		write_videopath = parser.get<std::string>("write_video");
	}
	if (k == 0) {
		frame = capture.open(0);
	}
	if (!capture.isOpened()){
		std::cout << "can't open" << std::endl;
		return -1;
	}
	NanoDet nanonet(320, 0.4, 0.5);

	size_t input_tensor_size_hrnet;

	std::map<const char*, std::vector<int64_t>> output_dim_hrnet;
	std::vector<const char*> input_node_names_hrnet;
	std::vector<const char*> output_node_names_hrnet;
	std::vector<int64_t> input_node_dims_hrnet;
	std::vector<int64_t> output_node_dims_hrnet;

	Ort::SessionOptions session_options;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	std::string model_hrnet_path = "s" + std::to_string(model) + ".onnx";
	std::wstring wstr;
	wstr.assign(model_hrnet_path.begin(), model_hrnet_path.end());
#ifdef _WIN32
	const wchar_t* model_path_hrnet = wstr.c_str();
#else
	const char* model_path_hrnet = model_hrnet_path.c_str();
#endif
	Ort::Session session_hrnet(env, model_path_hrnet, session_options);
	printf("Using Onnxruntime C++ API\n");
	Ort::AllocatorWithDefaultOptions allocator;

	input_tensor_size_hrnet = 1;
	// iterate over all input nodes
	for (int i = 0; i < session_hrnet.GetInputCount(); i++) {
		// print input node names
		char* input_name = session_hrnet.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);

		input_node_names_hrnet.push_back(input_name);

		// print input node types
		Ort::TypeInfo type_info = session_hrnet.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims_hrnet = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims_hrnet.size());
		for (int j = 0; j < input_node_dims_hrnet.size(); j++) {
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims_hrnet[j]);
			input_tensor_size_hrnet *= input_node_dims_hrnet[j];
		}
	}
	for (int i = 0; i < session_hrnet.GetOutputCount(); i++) {
		// print input node names
		char* output_name = session_hrnet.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);

		output_node_names_hrnet.push_back(output_name);

		// print input node types
		Ort::TypeInfo type_info = session_hrnet.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		output_node_dims_hrnet = tensor_info.GetShape();
		output_dim_hrnet[output_name] = output_node_dims_hrnet;
		printf("Output %d : num_dims=%zu\n", i, output_node_dims_hrnet.size());
		for (int j = 0; j < output_node_dims_hrnet.size(); j++) {
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims_hrnet[j]);
		}
	}
	static const std::string kWinName = "HRNet";
	cv::namedWindow(kWinName, cv::WINDOW_KEEPRATIO || cv::WINDOW_NORMAL);
	float* x_hrnet = new float[input_tensor_size_hrnet];
	std::vector<int> last_box_max = {0,0,0,0};
	cv::Size videoSize(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter writer(write_videopath, CV_FOURCC('M', 'J', 'P', 'G'), 24,videoSize);
	while (capture.read(frame)) {
		startTime_nano = clock();//计时开始
		std::vector<int> box_max{ 0,0,0,0 };
		nanonet.detect(frame, box_max);
		/*
		int area_box_max = (box_max[2] - box_max[0]) * (box_max[3] - box_max[1]);
		int area_last_box_max = (last_box_max[2] - last_box_max[0]) * (last_box_max[3] - last_box_max[1]);
		if (box_max[2] == 0 || area_box_max < 0.5 * area_last_box_max) {
			box_max = last_box_max;
		}
		else {
			last_box_max=box_max;
		}
		*/
		endTime_nano = clock();//计时结束
		std::cout << "The nanodet run time is: " << (double)(endTime_nano - startTime_nano) / CLOCKS_PER_SEC << "s" << std::endl;
		startTime_hrnet = clock();//计时开始
		
		std::vector<float> center{ 0,0 }, scale{ 0,0 };
		std::vector<int> img_size;
		if (model == 1) {
			img_size = { 288,384 };
		}
		else if (model == 4) {
			img_size = { 128,96 };
		}
		else{
			img_size = { 192,256 };
		}

		box_to_center_scale(box_max, img_size[0], img_size[1], center, scale);
		cv::Mat input;
		cv::Mat tran = get_affine_transform(center, scale, 0, img_size,0);
		cv::warpAffine(frame, input, tran, cv::Size(img_size[0],img_size[1]), cv::INTER_LINEAR);
		_normalize(input);

		convertMat2pointer(input, x_hrnet);
		auto memory_info_hrnet = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value input_tensor_hrnet = Ort::Value::CreateTensor<float>(memory_info_hrnet, x_hrnet, input_tensor_size_hrnet, input_node_dims_hrnet.data(), 4);

		// score model & input tensor, get back output tensor
		auto output_tensors_hrnet = session_hrnet.Run(Ort::RunOptions{ nullptr }, input_node_names_hrnet.data(), &input_tensor_hrnet, 1, output_node_names_hrnet.data(), 1);

		float* floatarr = output_tensors_hrnet[0].GetTensorMutableData<float>();
		float* preds = new float[output_dim_hrnet[output_node_names_hrnet[0]][1]*2+2];
		get_final_preds(floatarr, output_dim_hrnet[output_node_names_hrnet[0]], center, scale, preds);
		preds[34] = (preds[5] + preds[6]) / 2;
		preds[35] = (preds[5+17] + preds[+17]) / 2;
		int line_begin, line_end, iter, point_begin;
		if (t == 0) {//所有关节点
			line_begin = 0;
			line_end = 20;
			iter = 1;
			point_begin = 0;
		}
		//左侧
		else if (t == 1) {
			line_begin = 0;
			line_end = 10;
			iter = 2;
			point_begin = 0;
		}
		//右侧
		else if (t == 2) {
			line_begin = 10;
			line_end = 20;
			iter = 2;
			point_begin = 1;
		}
		rectangle(frame, cv::Point(box_max[0], box_max[1]), cv::Point(box_max[2], box_max[3]), cv::Scalar(0, 0, 255), 3);
		for (int i = line_begin; i < line_end; i = i + 2) {
			cv::line(frame, cv::Point2d(int(preds[pair_line[i]]), int(preds[pair_line[i] + 17])),
				cv::Point2d(int(preds[pair_line[i + 1]]), int(preds[pair_line[i + 1] + 17])), (0, 0, 255), 4);
		}

		for (int i = point_begin; i < 17; i=i+iter) {
			int x_coord = int(preds[i]);
			int y_coord = int(preds[i + 17]);
			cv::circle(frame, cv::Point2d(x_coord, y_coord), 1, (0, 255, 0), 2);
		}
		
		endTime_hrnet = clock();//计时结束
		std::cout << "The hrnet run time is: " << (double)(endTime_hrnet - startTime_hrnet) / CLOCKS_PER_SEC << "s" << std::endl;
		std::cout << "The run time is: " << (double)(endTime_hrnet - startTime_nano) / CLOCKS_PER_SEC << "s" << std::endl;
		std::string label = cv::format("%.2f", 1.0/(double)(endTime_hrnet - startTime_nano)*CLOCKS_PER_SEC);
		putText(frame, label+"FPS", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
		writer.write(frame);
		imshow(kWinName, frame);
		if (cv::waitKey(1)==27)
			break;
	}
	writer.release();
	cv::destroyAllWindows();
}