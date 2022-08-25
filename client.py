from jina import Client, Document

c = Client(host='grpc://0.0.0.0:51001')
# r = c.post('/create', Document())
# print(r.to_list())
# print(r.summary())
# r = c.post('/result', Document())
# print(r.to_list())


# da = c.post(
#     '/create',
#     parameters={
#         'sess_name': 'mydisco-123',
#         'prompt': 'hello',
#     },
# )

# check intermediate results
da = c.post('/result', parameters={'sess_name': 'mydisco-123'})
print(da.summary())
print(da.to_list())