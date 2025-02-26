import models.sin_model.model as SineModel
# import models.cos_model.model as CosModel

model = SineModel.SineModel()
model.train()
model.predict()
model.convert_to_tflite()

# model = CosModel.CosModel()
# model.train()
# model.predict()
# model.convert_to_tflite()
