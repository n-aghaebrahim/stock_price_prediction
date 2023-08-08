

#capitol = 210
#stock_current_price = 14.91

profit_percentage = [0.005, 0.01, 0.02, 0.03]

#for i in profit_percentage:
#    profit_taken = stock_current_price * i
#    stock_final_price = stock_current_price + profit_taken
    
#    print(f"The profit taken for {i} percentage is:\n$", profit_taken)
#    print(f"the final stock price for {i} percentage profit:\n$", stock_final_price)
#    print("---------------------------------------------------------\n")


# get the input
print('the available capitol to invest:\n')
available_capitol = float(input())

print('the average purchase price:\n')
average_price = float(input())

num_available_stock = available_capitol / average_price
print(f'********* you can purchase {num_available_stock} shares with respect to your available capitol ************')

print('the number of purchased share:\n')
num_share = float(input())

total_paid_price = num_share * average_price
print('total paid price: ', total_paid_price)

profit_percentage = [0.005, 0.01, 0.02, 0.03]

for i in profit_percentage:
    profit_taken = average_price * i
    stock_final_price = average_price + profit_taken
    total_final_shares_price = (average_price + profit_taken) * num_share
    
    print(f"\nThe profit taken for {i} percentage per stock is:\n$", profit_taken)
    print(f"the final stock price for {i} percentage profit:\n$", stock_final_price)
    print(f"the total shares profited price for {i} percentage is:\n$", total_final_shares_price)
    print(f"the total prifit is:\n$", (total_final_shares_price - total_paid_price))
    print("---------------------------------------------------------")
