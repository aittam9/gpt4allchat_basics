from gpt4all import GPT4All

#loading the model
model = GPT4All(model_name='orca-mini-3b.ggmlv3.q4_0.bin')

#chat function
def chat(model):
    #defining turns
    user_turn = "User> "
    assistant_turn = "Assistant> "
    
    # instantiate a chat loop
    while True:
        try:
            with model.chat_session():
                # gather user input
                user_input = input(user_turn)
                # use the input to generate a response
                response = model.generate(prompt = user_input, top_k= 1)
                #print the model response
                print(assistant_turn+model.current_chat_session[1]["content"])


        #exit the loop if user does keyboard interruption
        except KeyboardInterrupt:
                print(f"\n{assistant_turn}Bye, see you soon!")
                break

if __name__ == "__main__":
    chat(model) 