#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "sine_model.h"

#define IN_TEST false

#define BAUD_RATE 115200

constexpr float pi = 3.14159265;
constexpr float freq = 1;
constexpr float period = (1 / freq) * (1000);

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  
  constexpr int kTensorArenaSize = 32 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// --- functions and classes ---

void board_functionality_test_runner() {
  Serial.println("Testing board functionality.");
}

// --- main code ---

void setup() {
  Serial.begin(BAUD_RATE);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(
      error_reporter,
      "Model provided is schema version %d not equal to supported version %d.",
      model->version(), 
      TFLITE_SCHEMA_VERSION
    );
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
}

void loop() {
  if (IN_TEST) {
    board_functionality_test_runner();
    return;
  }

  unsigned long timestamp = millis();
  timestamp = timestamp % (unsigned long)period;

  float x_val = ((float)timestamp * 2 * pi) / period;
  model_input->data.f[0] = x_val;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
  }

  float y_val = model_output->data.f[0];
  delay(10);

  Serial.println(y_val);
}