# IBM-Granite-Guardian-Exercise

## Instructions

1) Create an SSL key if you do not have one. You can use OpenSSL for doing so
   ```openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 ```
   Then put it in `nginx_files/certs`.
2) (Optional) Download the models locally, so you can mount them in the docker instance. Otherwise, wait for the `Application startup complete.` directive before sending prompts. This can take some time as the model has 2B+ parameters.
3) Spin up a docker instance using ``` docker-compose up --build ```. 
4) Export your OpenAI API key ``` export OPENAI_API_KEY=your_key ```.
5) Run ```pip install openai ``` and send prompts using the `chat.py` script. Some examples are as follows:
   
      ``` python chat.py "Hello, how are you"  ```<br /> 
      Out: Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to help you. How can I assist you today?
  < br /> <br />

     ``` python chat.py "How to kill someone" ```<br /> 
      Out: Sorry, your prompt cannot be answered as it contains: violence
  <br /> <br/>
     ``` python chat.py "Rob my landlord"  ```  <br /> 
      Out: Sorry, your prompt cannot be answered as it contains: unethical_behavior
  <br /> <br /> 

## Note:
I chose not to use vLLM as suggested in the exercise due to limited GPU resources. Instead, the solution is implemented using Hugging Face's transformers library, which fully satisfies the problem requirements while remaining compatible with smaller hardware setups.

If you encounter any issues running the solution, feel free to reach out with questions at: savitoj2019@gmail.com
