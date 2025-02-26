/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/



#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// Mảng các mẫu test cho các chữ số khác nhau
const float test_samples[][8][8] = {
    // Số 0
    {
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.498039f, 0.498039f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.498039f, 0.498039f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f}
    },
    
    // Số 1
    {
        {0.000000f, 0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.749020f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f}
    },

    // Số 2 (mẫu hiện tại của bạn)
    {
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.749020f, 0.200000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.749020f, 0.200000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.498039f, 0.200000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.498039f, 0.200000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.200000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.247059f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.247059f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.247059f, 0.000000f, 0.000000f}
    },

    // Số 3
    {
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f}
    },

    // Số 5
    {
        {0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.749020f, 0.749020f, 0.000000f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.498039f, 0.749020f, 0.000000f, 0.000000f, 0.749020f, 0.498039f, 0.000000f},
        {0.000000f, 0.000000f, 0.498039f, 0.749020f, 0.749020f, 0.498039f, 0.000000f, 0.000000f},
        {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f}
    }
};

// Nhãn thực tế tương ứng với mỗi mẫu
const int true_labels[] = {0, 1, 2, 3, 5};
// Số lượng mẫu test
const int NUM_SAMPLES = sizeof(true_labels) / sizeof(true_labels[0]);

// Model được include dưới dạng hex array
#include "digit_model_NCKH.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;


// Define the size of the input and output tensors
constexpr int kInputTensorSize = 2;
constexpr int kOutputTensorSize = 1;

// Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 64* 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 TensorFlow Lite Test");

  // Set up logging (will report to Serial, even within TFLite functions)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure
    model = tflite::GetModel(digit_model_NCKH);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      TF_LITE_REPORT_ERROR(error_reporter,
                          "Model provided is schema version %d not equal "
                          "to supported version %d.",
                          model->version(), TFLITE_SCHEMA_VERSION);
      return;
    }

      // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }

    // Lấy con trỏ đến input và output tensors
    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    Serial.println("Model đã được load thành công");
    // Serial.print("Input shape: ");
    // Serial.print(input->dims->data[1]);
    // Serial.print(" x ");
    // Serial.print(input->dims->data[2]);
    // Serial.print(" x ");
    // Serial.println(input->dims->data[3]);
}

// Trong hàm loop(), thay đổi để test lần lượt các mẫu:
void loop() {
    static int current_sample = 0;

    // In thông tin mẫu hiện tại
    Serial.printf("\n\nTest mẫu %d (Nhãn thực tế: %d)\n", 
                 current_sample, true_labels[current_sample]);

    // Copy mẫu test vào input tensor
    float* input_data = model_input->data.f;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            input_data[i * 8 + j] = test_samples[current_sample][i][j];
        }
    }

    // Thực hiện inference
    long start_time = millis();
    TfLiteStatus invoke_status = interpreter->Invoke();
    long end_time = millis();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return;
    }

    // Lấy kết quả
    float* output_data = model_output->data.f;
    
    // Tìm chữ số có xác suất cao nhất
    float max_prob = 0;
    int predicted_digit = -1;
    for (int i = 0; i < 10; i++) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            predicted_digit = i;
        }
    }

    // In kết quả
    Serial.println("\nKết quả dự đoán:");
    Serial.printf("Chữ số thực tế: %d\n", true_labels[current_sample]);
    Serial.printf("Chữ số dự đoán: %d\n", predicted_digit);
    Serial.printf("Độ tin cậy: %.2f%%\n", max_prob * 100);
    Serial.printf("Thời gian: %ldms\n", end_time - start_time);

    // In xác suất cho tất cả các chữ số
    Serial.println("\nXác suất cho từng chữ số:");
    for (int i = 0; i < 10; i++) {
        Serial.printf("%d: %.2f%%\n", i, output_data[i] * 100);
    }

    // Chuyển sang mẫu tiếp theo
    current_sample = (current_sample + 1) % NUM_SAMPLES;

    delay(5000); // Đợi 5 giây trước khi test mẫu tiếp theo
}

