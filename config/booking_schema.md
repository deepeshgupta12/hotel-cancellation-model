# Booking Table Schema (Source of Truth)

This defines the minimum fields we expect for the hotel booking cancellation model.
Real production data can have more columns, but these are the core ones.

| Column name           | Type        | Example                      | Description |
|-----------------------|------------|------------------------------|-------------|
| booking_id            | string     | BKG-10001                    | Unique booking identifier. |
| user_id               | string     | U-1234                       | Unique user/customer ID (or hashed). |
| hotel_id              | string     | H-99                         | Unique hotel ID. |
| booking_datetime      | datetime   | 2025-01-03T10:15:00          | Timestamp when booking was created. |
| checkin_date          | date       | 2025-02-10                   | Planned check-in date. |
| checkout_date         | date       | 2025-02-13                   | Planned check-out date. |
| booking_channel       | string     | web / app / ota / phone      | Channel used to make the booking. |
| device_type           | string     | mobile / desktop             | Device category at booking time. |
| rate_plan             | string     | refundable / non_refundable  | Rate plan type. |
| payment_status        | string     | prepaid / pay_at_hotel       | Payment type/status at booking time. |
| booking_amount        | float      | 14500.0                      | Total booking amount (in base currency). |
| currency              | string     | INR / USD                    | Currency code. |
| num_guests            | int        | 2                            | Total number of guests. |
| num_rooms             | int        | 1                            | Number of rooms booked. |
| user_country          | string     | IN / US                      | ISO country code of user or billing. |
| status                | string     | confirmed / cancelled        | Final booking status, if known. |
| is_cancelled          | int (0/1)  | 0 / 1                        | Label: 1 if booking cancelled or no-show by check-in date. |
| cancellation_datetime | datetime   | 2025-02-05T18:30:00          | When the booking was cancelled (if applicable). |
| no_show_flag          | int (0/1)  | 0 / 1                        | 1 if user did not show up on check-in date without formal cancellation. |

Notes:
- In real data, `is_cancelled` will often be derived from `status`, `cancellation_datetime` and `no_show_flag`.
- For our initial experiments, we will keep both `status` and `is_cancelled` in the dataset for clarity.
