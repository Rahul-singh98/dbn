FROM test:latest

RUN pip3 install pandas matplotlib scikit-learn pickle

COPY . /app

WORKDIR /app

# RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD ['main.py' ]