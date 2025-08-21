def get_valid_input_str(prompt: str, valid_options: list[str]) -> str:
    while True:
        value = input(prompt)
        if value in valid_options:
            return value
        else:
            print(f"❌ Invalid input. Please enter one of: {', '.join(valid_options)}")

def get_valid_input_int(prompt: str,min,max):
    while True:
        value = input(prompt)
        value_int=int(value)
        if min <= value_int <= max:
            return value_int
        else:
            print(f"Invalid input. Please enter a value between {min} and {max}")

def get_valid_input_float(prompt: str, min: float, max: float):
    while True:
        value = input(prompt)
        value_int=float(value)
        if min <= value_int <= max:
            return value_int
        else:
            print(f"Invalid input. Please enter a value between {min} and {max}")    

