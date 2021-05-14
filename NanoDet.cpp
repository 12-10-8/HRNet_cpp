#include "NanoDet.h"
NanoDet::NanoDet(int input_shape, float confThreshold, float nmsThreshold)
{
	assert(input_shape == 320 || input_shape == 416);
	this->input_shape[0] = input_shape;
	this->input_shape[1] = input_shape;
	this->prob_threshold = confThreshold;
	this->iou_threshold = nmsThreshold;

	std::ifstream ifs(this->classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) this->classes.push_back(line);
	this->num_class = this->classes.size();
}

Ort::Session NanoDet::load_onnx(){
	Ort::SessionOptions session_options;
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#ifdef _WIN32
	const wchar_t* model_path = L"nanodet.onnx";
#else
	const char* model_path = "nanodet.onnx";
#endif
	Ort::Session session(env, model_path, session_options);
	printf("Using Onnxruntime C++ API\n");
	Ort::AllocatorWithDefaultOptions allocator;
	input_tensor_size = 1;
	// iterate over all input nodes
	for (int i = 0; i < session.GetInputCount(); i++) {
		// print input node names
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);

		input_node_names.push_back(input_name);

		// print input node types
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++) {
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
			input_tensor_size *= input_node_dims[j];
		}
	}
	for (int i = 0; i < session.GetOutputCount(); i++) {
		// print input node names
		char* output_name = session.GetOutputName(i, allocator);
		printf("Output %d : name=%s\n", i, output_name);

		output_node_names.push_back(output_name);

		// print input node types
		Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Output %d : type=%d\n", i, type);

		output_node_dims = tensor_info.GetShape();
		output_dim[output_name] = output_node_dims;
		printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
		for (int j = 0; j < output_node_dims.size(); j++) {
			printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
		}
	}
	return session;
}

void NanoDet::detect(cv::Mat frame, std::vector<int>& box_max) {
	int newh = 0, neww = 0, top = 0, left = 0;
	cv::Mat dstimg = resize_image(frame, &newh, &neww, &top, &left);
	_normalize(dstimg);
	float* x = new float[input_tensor_size];
	convertMat2pointer(dstimg, x);
	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, x, input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors = this->session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 6);
	assert(output_tensors.size() == 6 && output_tensors.front().IsTensor());

	// Get pointer to output tensor float values
	std::vector<cv::Mat> outs;
	for (int i = 0; i < 6; ++i) {
		float* floatarr = output_tensors[i].GetTensorMutableData<float>();
		outs.push_back(Array2Mat(floatarr, output_dim[output_node_names[i]]));
	}
	post_process(outs, frame, newh, neww, top, left, box_max);
}

cv::Mat NanoDet::resize_image(cv::Mat& srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->input_shape[0];
	*neww = this->input_shape[1];
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->input_shape[0];
			*neww = int(this->input_shape[1] / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->input_shape[1] - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->input_shape[1] - *neww - *left, cv::BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->input_shape[0] * hw_scale;
			*neww = this->input_shape[1];
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->input_shape[0] - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->input_shape[0] - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}




void NanoDet::softmax(float* x, int length)
{
	float sum = 0;
	int i = 0;
	for (i = 0; i < length; i++)
	{
		x[i] = exp(x[i]);
		sum += x[i];
	}
	for (i = 0; i < length; i++)
	{
		x[i] /= sum;
	}
}

void NanoDet::generate_proposal(std::vector<int>& classIds, std::vector<float>& confidences, std::vector<cv::Rect>& boxes, const int stride_, cv::Mat out_score, cv::Mat out_box)
{
	const int num_grid_y = (int)this->input_shape[0] / stride_;
	const int num_grid_x = (int)this->input_shape[1] / stride_;
	const int reg_1max = this->reg_max + 1;

	if (out_score.dims == 3)
	{
		out_score = out_score.reshape(0, num_grid_x * num_grid_y);
	}
	if (out_box.dims == 3)
	{
		out_box = out_box.reshape(0, num_grid_x * num_grid_y);
	}
	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const int idx = i * num_grid_x + j;
			cv::Mat scores = out_score.row(idx).colRange(0, num_class);
			cv::Point classIdPoint;
			double score;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &score, 0, &classIdPoint);
			if (score >= this->prob_threshold)
			{
				float* pbox = (float*)out_box.data + idx * reg_1max * 4;
				float dis_pred[4];
				for (int k = 0; k < 4; k++)
				{
					this->softmax(pbox, reg_1max);
					float dis = 0.f;
					for (int l = 0; l < reg_1max; l++)
					{
						dis += l * pbox[l];
					}
					dis_pred[k] = dis * stride_;
					pbox += reg_1max;
				}

				float pb_cx = (j + 0.5f) * stride_ - 0.5;
				float pb_cy = (i + 0.5f) * stride_ - 0.5;
				float x0 = pb_cx - dis_pred[0];
				float y0 = pb_cy - dis_pred[1];
				float x1 = pb_cx + dis_pred[2];
				float y1 = pb_cy + dis_pred[3];

				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
				boxes.push_back(cv::Rect((int)x0, (int)y0, (int)(x1 - x0), (int)(y1 - y0)));
			}
		}
	}
}

void NanoDet::post_process(std::vector<cv::Mat>& outs, cv::Mat& frame, int newh, int neww, int top, int left, std::vector<int>& box_max)
{
	/////generate proposals
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	this->generate_proposal(classIds, confidences, boxes, this->stride[0], outs[0], outs[3]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[1], outs[1], outs[4]);
	this->generate_proposal(classIds, confidences, boxes, this->stride[2], outs[2], outs[5]);

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->prob_threshold, this->iou_threshold, indices);
	float ratioh = (float)frame.rows / newh;
	float ratiow = (float)frame.cols / neww;
	int max_area = -10;
	int x1, y1, x2, y2;
	std::string label1;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		int xmin = (int)std::max((box.x - left) * ratiow, 0.f);
		int ymin = (int)std::max((box.y - top) * ratioh, 0.f);
		int xmax = (int)std::min((box.x - left + box.width) * ratiow, (float)frame.cols);
		int ymax = (int)std::min((box.y - top + box.height) * ratioh, (float)frame.rows);
		std::string label = cv::format("%.2f", confidences[idx]);
		label = classes[classIds[idx]] + ":" + label;
		if (box.width * box.height > max_area && classes[classIds[idx]].compare("person") == 0) {
			max_area = box.width * box.height;
			x1 = xmin;
			y1 = ymin;
			x2 = xmax;
			y2 = ymax;
			label1 = label;
		}
		//Display the label at the top of the bounding box
		int baseLine;
		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		ymin = std::max(ymin, labelSize.height);
	}
	if (max_area != -10) {
		box_max[0] = x1;
		box_max[1] = y1;
		box_max[2] = x2;
		box_max[3] = y2;
	}
}