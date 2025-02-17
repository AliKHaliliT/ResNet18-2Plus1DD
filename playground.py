import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from ResNet182Plus1DD import ResNet182Plus1DD


model = ResNet182Plus1DD()
model.build((None, 32, 256, 256, 3))
model.summary()