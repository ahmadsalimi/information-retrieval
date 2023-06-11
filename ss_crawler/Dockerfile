FROM selenium/standalone-chrome

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
