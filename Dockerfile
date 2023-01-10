FROM public.ecr.aws/lambda/python:3.9

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
RUN pip install numpy Pillow

COPY intel-classifier.tflite .
COPY lambda_func.py .

CMD [ "lambda_func.lambda_handler" ]