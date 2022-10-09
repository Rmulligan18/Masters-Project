
#include <Arduino_LSM9DS1.h>
#include <Arduino_APDS9960.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino.h>


#include "nonoptmodel.h"



const int ledPin1 = 22;
const int ledPin2 = 23;
const int ledPin3 = 24;

const float accelerationThreshold = 2; // 
const int numSamples = 239;

int samplesRead = numSamples;

tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURE[] = {
  "hand to mouth gesture", "non-hand to mouth gesture"
};

#define NUM_GESTURE (sizeof(GESTURE) / sizeof(GESTURE[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();
  Serial.println("Perform a gesture to begin classification");
  Serial.println();

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);

  }
  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  pinMode(24, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(ledPin1, HIGH); // red
  digitalWrite(ledPin2, HIGH); // green
  digitalWrite(ledPin3, HIGH); // blue
  Serial.begin(9600);
  while (!Serial);

  //if (!APDS.begin()) {
    //Serial.println("Error initializing APDS9960 sensor!");
  //}

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);

      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

      // check if it's above the threshold
      if (aSum >= accelerationThreshold) {
        // reset the sample read count

        samplesRead = 0;
        break;
      }
    }
  }
  
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

       for (int i = 0; i < NUM_GESTURE; i++) {
          Serial.print(GESTURE[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 6);

        if(GESTURE[i]==GESTURE[1] && tflOutputTensor->data.f[i]<0.5)
        {
          digitalWrite(LEDG,LOW);
          delay(1000);
          digitalWrite(LEDG,HIGH);
          delay(1000);              // wait for a second
          Serial.print("Detected gesture: hand to mouth. Accuracy: "  );
          Serial.println(tflOutputTensor->data.f[0], 6);

          
          break;
        }
        
        else if(GESTURE[i]==GESTURE[1] && tflOutputTensor->data.f[i]>0.5)
        {
          digitalWrite(LEDB,LOW);
          delay(1000);
          digitalWrite(LEDB,HIGH);
          delay(1000);              // wait for a second
          Serial.print("Detected gesture: non-hand to mouth. Accuracy: " );
          Serial.println(tflOutputTensor->data.f[1], 6);

          break;
        }
        }
          Serial.println();
      }
    }
  }
}
