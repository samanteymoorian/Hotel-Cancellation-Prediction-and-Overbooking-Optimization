# Hotel Cancellation Prediction and Overbooking Optimization

Hotel cancellations create uncertainty in room availability and revenue. Accepting too few reservations may leave rooms unoccupied, while accepting too many creates the risk of having more guests arrive than the hotel can accommodate.

This project investigates whether customer-level cancellation predictions can improve overbooking decisions compared with relying on historical cancellation rates.

The analysis combines machine learning with a discrete optimization model. Several classification algorithms are evaluated to estimate cancellation probabilities, which are subsequently used to determine the number of additional reservations that maximizes expected profit.

## Data and preparation

I use the [Hotel Booking Demand dataset]("https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"), available through TidyTuesday. To keep the booking decision manageable, I restrict the analysis to City Hotel reservations for **one night**, with at least one adult. This leaves **13,227 reservations**.

The preparation includes examining missing values and unusual room rates, creating indicators for whether an agent or company ID is recorded, and extracting arrival-date features. For cancellation prediction, I use 14 booking-related features, including lead time, previous booking history, guest counts, market segment, distribution channel, and reserved room type. Outcome-related fields such as `reservation_status` and `reservation_status_date` are excluded.

Rather than randomly mixing bookings across years, I split the data chronologically by arrival date:

| Split | Reservations | Purpose |
| --- | ---: | --- |
| Training | 8,807 | Fit the models |
| Validation | 1,505 | Tune parameters and choose a model |
| Test | 2,915 | Evaluate the selected model |

Scaling and one-hot encoding are handled inside scikit-learn pipelines, with the preprocessing fitted only on the training data during model development.

## Predicting cancellations

I compare Logistic Regression, Random Forest, and LightGBM against a baseline that predicts the same historical cancellation rate for every reservation. I tune the three ML models on the validation set, using **Brier score** as the main selection metric. Since the optimization step uses probabilities, their quality matters more here than classification accuracy at a fixed threshold.

### Validation results

| Model | Brier score (lower is better) |
| --- | ---: |
| **Logistic Regression** | **0.1805** |
| Random Forest | 0.1810 |
| Historical cancellation rate | 0.2031 |
| LightGBM | 0.2070 |

Logistic Regression performed best on validation, though Random Forest was close. I refitted the selected Logistic Regression model on the combined training and validation periods before evaluating it on the test set.

### Held-out test results

| Model | ROC-AUC | Brier score | Log loss |
| --- | ---: | ---: | ---: |
| Historical cancellation rate | 0.5000 | 0.2276 | 0.6482 |
| **Logistic Regression** | **0.7612** | **0.1802** | **0.5349** |

I also inspect reliability curves to see whether predicted cancellation probabilities match observed cancellation frequencies. I use the probabilities directly; I have not fitted a separate post-hoc calibration model.

## From predictions to overbooking decisions

For the decision experiment, I assume a hotel with **10 rooms** and allow up to **15 additional reservations**, subject to the number of candidate bookings available for an arrival date. The illustrative economics are **105 per occupied room** and a **230 penalty per arriving guest who cannot be accommodated**.

I compare three policies:

- **No overbooking:** Accept at most 10 reservations.
- **Historical-rate optimization:** Choose an overbooking limit using the overall cancellation rate estimated from development data.
- **ML-based optimization:** Choose the limit using the predicted cancellation probability of each candidate reservation.

For each feasible limit, I calculate the probability distribution of the number of guests arriving, assuming independent cancellation outcomes. This gives an expected profit for every possible decision. The optimizer simply selects the integer limit with the highest expected profit. The policies can then be compared using the cancellation outcomes in the held-out test period.

## Overbooking Optimization Results

The optimization model was evaluated on 111 eligible arrival dates from the held-out test set. Three policies were compared: no overbooking, optimization based on the historical cancellation rate, and optimization using customer-specific cancellation probabilities from Logistic Regression.

For each date, the model selected an overbooking limit that maximized expected profit. The decisions were then evaluated using observed cancellation outcomes and the assumed room revenue and overbooking penalties.

### Comparison of Overbooking Strategies

| Metric | No Overbooking | Historical Rate | ML Optimization |
|---|---:|---:|---:|
| Average daily simulated profit | 564.73 | 685.14 | **716.53** |
| Average room occupancy | 53.78% | 71.17% | **82.25%** |
| Average additional bookings | 0.00 | 2.78 | 5.22 |
| Average bumped guests per day | 0.00 | 0.27 | 0.64 |

![Average Simulated Profit by Overbooking Strategy](Average%20Simulated%20Profit%20by%20Overbooking%20Strategy.png)

The ML-based policy achieved the highest average realized simulated profit, outperforming historical-rate optimization by **4.6%** and the no-overbooking strategy by **26.9%**.

The additional profit was accompanied by higher room occupancy. However, ML optimization also resulted in more guests exceeding hotel capacity: an average of 0.64 per day, compared with 0.27 under historical-rate optimization. This illustrates the trade-off between increasing occupancy and the cost of overbooking.

### Value of Individualized Cancellation Predictions

To evaluate the contribution of machine learning, daily realized profits from ML-based optimization were compared directly with those from historical-rate optimization.

The results show that:

- **40.5%** of days: ML optimization performed better.
- **37.8%** of days: Both policies generated equal profit.
- **21.6%** of days: Historical-rate optimization performed better.

Overall, ML-based optimization performed **at least as well as historical-rate optimization on 78.4% of evaluated days**.

The average simulated profit improvement was 31.40 per day, corresponding to 3,485 across the 111 eligible days.

These findings suggest that individualized cancellation probabilities can improve overbooking decisions compared with relying on a single historical cancellation rate. The improvement was not consistent across every date, however, and the ML-based strategy sometimes increased the number of guests who could not be accommodated.

### Interpretation and Limitations

The results demonstrate how cancellation predictions can inform operational decisions rather than being used solely for classification.

The financial improvements are simulated and depend on the assumed hotel capacity, room revenue, overbooking penalty, and independence of cancellation outcomes. They should not be interpreted as actual revenue gains achieved by the hotel.

## Running the notebook

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm notebook
```

Then open the notebook:

```bash
jupyter notebook "Hotel Cancellation Prediction and Overbooking Optimization.ipynb"
```

The notebook downloads the public CSV directly from GitHub, so there is no separate local data file to prepare. Run the cells from top to bottom.

## Limitations

This is a **simulation of an overbooking decision**, not a reconstruction of a hotel's actual revenue management system. In particular:

- Hotel capacity, room revenue, and bumping penalties are assumed rather than observed.
- The dataset contains recorded reservations, not rejected booking requests or a complete booking-acceptance history. Booking order is approximated using lead time.
- The experiment uses one-night reservations only and treats non-canceled bookings as arrivals.
- The arrival distribution assumes independent cancellations.
- Some recorded features may have been updated after the original booking was made, so the historical data cannot perfectly reproduce booking-time information.
- Although the dataset includes ADR and deposit type, the current optimization uses a fixed room-revenue assumption and does not model cancellation-fee income.

The aim is to examine whether individualized cancellation probabilities can improve a clearly defined decision rule, while keeping the assumptions and limitations visible.

## Data source

[Hotel Booking Demand — TidyTuesday (2020-02-11)](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11)
