import sys
def model_test(model , k):
    try : 
        print(model(k).size())
        print("Model Creating success ")

    except :
        print("failed")
        raise
    try :
        model.info(sys.stdout)
    except :
        print("Lazy...")