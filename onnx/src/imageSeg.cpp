#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <onnxruntime_cxx_api.h>


#include<opencv4/opencv2/core.hpp>
#include<opencv4/opencv2/highgui.hpp>
#include<opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

bool CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        std::cerr << msg << std::endl;
        g_ort->ReleaseStatus(status);
        throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
    }
    return true;
}

// ͼ����  ��׼������
void PreProcess2(const Mat& image, Mat& image_blob)
{
    Mat input;
    image.copyTo(input);

    //���ݴ��� ��׼��
    std::vector<Mat> channels, channel_p;
    split(input, channels);
    Mat R, G, B;
    B = channels.at(0);
    G = channels.at(1);
    R = channels.at(2);

    B = B / 255.0;
    G = G / 255.0;
    R = R / 255.0;

    channel_p.push_back(R);
    channel_p.push_back(G);
    channel_p.push_back(B);

    Mat outt;
    merge(channel_p, outt);
    image_blob = outt;
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


void run_ort_snpe_ep(std::string backend, std::string input_path) {
#ifdef _WIN32
    const wchar_t* model_path = L"data/torch.onnx";
#else
    const char* model_path = "data/torch.onnx";
#endif

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env;
    CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    OrtSessionOptions* session_options;
    CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
    CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
    CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC));

    std::vector<const char*> options_keys = { "runtime", "buffer_type" };
    std::vector<const char*> options_values = { backend.c_str(), "FLOAT" };  // set to TF8 if use quantized data

    //CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider(session_options, "SNPE", options_keys.data(),
    //    options_values.data(), options_keys.size()));

    OrtSession* session;
    CheckStatus(g_ort, g_ort->CreateSession(env, model_path, session_options, &session));

    OrtAllocator* allocator;
    CheckStatus(g_ort, g_ort->GetAllocatorWithDefaultOptions(&allocator));
    size_t num_input_nodes;
    CheckStatus(g_ort, g_ort->SessionGetInputCount(session, &num_input_nodes));

    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<ONNXTensorElementDataType> input_types;
    std::vector<OrtValue*> input_tensors;

    input_node_names.resize(num_input_nodes);
    input_node_dims.resize(num_input_nodes);
    input_types.resize(num_input_nodes);
    input_tensors.resize(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Get input node names
        char* input_name;
        CheckStatus(g_ort, g_ort->SessionGetInputName(session, i, allocator, &input_name));
        input_node_names[i] = input_name;

        std::cout << "input name :" << input_name  <<std::endl;

        // Get input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort, g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort, g_ort->GetTensorElementType(tensor_info, &type));
        input_types[i] = type;

        // Get input shapes/dims
        size_t num_dims;
        CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
        input_node_dims[i].resize(num_dims);
        CheckStatus(g_ort, g_ort->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims));

        std::cout << "input dims :" << num_dims << std::endl;

        size_t tensor_size;
        CheckStatus(g_ort, g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

        if (typeinfo) g_ort->ReleaseTypeInfo(typeinfo);
    }

    size_t num_output_nodes;
    std::vector<const char*> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims;
    std::vector<OrtValue*> output_tensors;
    CheckStatus(g_ort, g_ort->SessionGetOutputCount(session, &num_output_nodes));
    output_node_names.resize(num_output_nodes);
    output_node_dims.resize(num_output_nodes);
    output_tensors.resize(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        // Get output node names
        char* output_name;
        CheckStatus(g_ort, g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        output_node_names[i] = output_name;

        std::cout << "output dims :" << output_name << std::endl;

        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort, g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

        // Get output shapes/dims
        size_t num_dims;
        CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
        output_node_dims[i].resize(num_dims);
        CheckStatus(g_ort, g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));

        std::cout << "output dims :" << num_dims << std::endl;

        size_t tensor_size;
        CheckStatus(g_ort, g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

        if (typeinfo) g_ort->ReleaseTypeInfo(typeinfo);
    }

    
    //size_t input_data_size = 1 * 500 * 500 * 3;
    //size_t input_data_length = input_data_size * sizeof(float);
    //std::vector<float> input_data(input_data_size, 1.0);

    //std::ifstream input_raw_file(input_path, std::ios::binary);
    //input_raw_file.seekg(0, std::ios::end);
    //const size_t num_elements = input_raw_file.tellg() / sizeof(float);
    //input_raw_file.seekg(0, std::ios::beg);
    //input_raw_file.read(reinterpret_cast<char*>(&input_data[0]), num_elements * sizeof(float));

    //����ͼƬ
    Mat img = imread(input_path);
    Mat det1;
    //resize(img, det1, Size(500, 500), INTER_AREA);

    img.convertTo(img, CV_32FC3);
    PreProcess2(img, det1);         //��׼������
;

    Mat blob = dnn::blobFromImage(det1, 1., Size(500, 500), Scalar(0, 0, 0), false, false);
    printf("Load success!\n");

    /*
    //pre-processing the Image
    // step 1: Read an image in HWC BGR UINT8 format.
    cv::Mat imageBGR = cv::imread(input_path, cv::ImreadModes::IMREAD_COLOR);

    //// step 2: Resize the image.
    cv::Mat resizedImageRGB, resizedImage, preprocessedImage;

    //// step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(imageBGR, resizedImageRGB,
        cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    //// step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = 1 * 500 * 500 * 3;
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
    preprocessedImage.end<float>());
    */

    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    /*CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, reinterpret_cast<void*>(input_data.data()), input_data_length,
        input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
    g_ort->ReleaseMemoryInfo(memory_info);*/

    //CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(memory_info, inputTensorValues.data(), inputTensorSize * sizeof(float), input_node_dims[0].data(),
    //            input_node_dims[0].size(), input_types[0], &input_tensors[0]));

    CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(memory_info, blob.ptr<float>(), blob.total() * sizeof(float), input_node_dims[0].data(),
        input_node_dims[0].size(), input_types[0], &input_tensors[0]));

    CheckStatus(g_ort, g_ort->Run(session, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
        input_tensors.size(), output_node_names.data(), output_node_names.size(),
        output_tensors.data()));

    size_t output_data_size = 500 * 500;
    size_t output_data_length = output_data_size * sizeof(int64_t*);
    std::vector<int64_t*> output_data(output_data_length);
    void* output_buffer;
    CheckStatus(g_ort, g_ort->GetTensorMutableData(output_tensors[0], &output_buffer));
    int64_t* int_buffer = reinterpret_cast<int64_t*>(output_buffer);

   /* auto max = std::max_element(int_buffer, int_buffer + output_data_size);
    int max_index = static_cast<int>(std::distance(int_buffer, max));*/

    //std::cout << max_index << std::endl;
    
    int count = 0;
    Mat newarr = Mat_<int>(500, 500); //����һ��500*500�ľ���
    for (int i = 0; i < newarr.rows; i++)
    {
        for (int j = 0; j < newarr.cols; j++) //��������ѭ��
        {
           if ((int)int_buffer[i * j + j] == 1) {
               count++;
               newarr.at<int>(i, j) = 255;
               continue;
            }
           newarr.at<int>(i, j) = int_buffer[i * j + j];
       }
    }
    cout << count << endl;

    imwrite("./test.png", newarr);
    newarr = imread("./test.png", IMREAD_GRAYSCALE);
    cout << newarr.channels() << endl;
    //imshow("mask", newarr);
    

    //cv::imshow("mask", newarr);
    //cv::waitKey();
}

void PrintHelp() {
    std::cout << "To run the sample, use the following command:" << std::endl;
    std::cout << "Example: ./snpe_ep_sample --cpu <path_to_raw_input>" << std::endl;
    std::cout << "To Run with SNPE CPU backend. Example: ./snpe_ep_sample --cpu chairs.raw" << std::endl;
    std::cout << "To Run with SNPE DSP backend. Example: ./snpe_ep_sample --dsp chairs.raw" << std::endl;
}

constexpr const char* CPUBACKEDN = "--cpu";
constexpr const char* DSPBACKEDN = "--dsp";

int main() {
    /*std::string backend = "CPU";

    if (argc != 3) {
        PrintHelp();
        return 1;
    }

    if (strcmp(argv[1], CPUBACKEDN) == 0) {
        backend = "CPU";
    }
    else if (strcmp(argv[1], DSPBACKEDN) == 0) {
        backend = "DSP";
    }
    else {
        std::cout << "This sample only support CPU, DSP." << std::endl;
        PrintHelp();
        return 1;
    }
    std::string input_path(argv[2]);*/
    std::string backend = "CPU";
    std::string input_path = "data/1.jpg";
    run_ort_snpe_ep(backend, input_path);
    return 0;
}