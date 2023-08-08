def make_decision(prediction_results: dict):
    buy_track = 0
    sell_track = 0

    prob_b = []
    prob_s = []
    track_b = []
    track_s = []
    prob_total = []

    track_b_less10 = []
    track_b_more10 = []
    track_s_less10 = []
    track_s_more10 = []


    for k, v in prediction_results.items():
        if k < 4:
            close_less10 = float(v['Close'])
            open_less10 = float(v['Open'])
            diff_less10 = close_less10 - open_less10

            if diff_less10>0:
                prob_less10 = diff_less10
                track_b_less10.append(prob_less10)

            else:
                prob_less10 = diff_less10
                track_s_less10.append(prob_less10)

        elif k>10:
            close_more10 = float(v['Close'])
            open_more10 = float(v['Open'])
            diff_more10 = close_more10 - open_more10

            if diff_more10>0:
                prob_more10 = diff_more10
                track_b_more10.append(prob_more10)

            else:
                prob_more10 = diff_more10
                track_s_more10.append(prob_more10)


    total_b_less10 = sum(track_b_less10)
    total_s_less10 = sum(track_s_less10)

    total_b_more10 = sum(track_b_more10)
    total_s_more10 = sum(track_s_more10)
    print('total b less 10: ', total_b_less10)
    print('total s less 10: ', total_s_less10)
    print('total b more 5: ', total_b_more10)
    print('total s less 5: ', total_s_more10)





    for k, v in prediction_results.items():
        close = float(v["Close"])
        openn = float(v["Open"])
        diff = close - openn
        if diff > 0:
            prob = diff
            track_b.append(prob)
        else:
            prob = diff
            track_s.append(prob)

    total_b = sum(track_b)
    total_s = sum(track_s)
    print(total_s)
    print(total_b)
    total = total_b + total_s
    print(total)

    if total > 0:
        d = str(total) + "B"

    else:
        d = str(total) + "S"

    print(
        f"*********************************  the final desicion is:------------->>>> {d}"
    )


