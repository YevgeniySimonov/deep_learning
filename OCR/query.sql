SELECT r.receipt_identifier as "receiptId", r.content as "content"
FROM receipt_version r
WHERE number_of_pages = 1
LIMIT 5;