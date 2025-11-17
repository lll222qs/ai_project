# # # # # # # # # # # # # from typing import Annotated

# # # # # # # # # # # # # from fastapi import Depends, FastAPI

# # # # # # # # # # # # # app = FastAPI()


# # # # # # # # # # # # # async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
# # # # # # # # # # # # #     return {"q": q, "skip": skip, "limit": limit}


# # # # # # # # # # # # # @app.get("/items/")
# # # # # # # # # # # # # async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
# # # # # # # # # # # # #     return commons


# # # # # # # # # # # # # @app.get("/users/")
# # # # # # # # # # # # # async def read_users(commons: Annotated[dict, Depends(common_parameters)]):
# # # # # # # # # # # # #     return commons



# # # # # # # # # # # # from typing import Annotated

# # # # # # # # # # # # from fastapi import Depends, FastAPI

# # # # # # # # # # # # app = FastAPI()


# # # # # # # # # # # # async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
# # # # # # # # # # # #     return {"q": q, "skip": skip, "limit": limit}


# # # # # # # # # # # # CommonsDep = Annotated[dict, Depends(common_parameters)]


# # # # # # # # # # # # @app.get("/items/")
# # # # # # # # # # # # async def read_items(commons: CommonsDep):
# # # # # # # # # # # #     return commons


# # # # # # # # # # # # @app.get("/users/")
# # # # # # # # # # # # async def read_users(commons: CommonsDep):
# # # # # # # # # # # #     return commons



# # # # # # # # # # # from fastapi import BackgroundTasks, FastAPI

# # # # # # # # # # # app = FastAPI()


# # # # # # # # # # # def write_notification(email: str, message=""):
# # # # # # # # # # #     with open("log.txt", mode="w") as email_file:
# # # # # # # # # # #         content = f"notification for {email}: {message}"
# # # # # # # # # # #         email_file.write(content)


# # # # # # # # # # # @app.post("/send-notification/{email}")
# # # # # # # # # # # async def send_notification(email: str, background_tasks: BackgroundTasks):
# # # # # # # # # # #     background_tasks.add_task(write_notification, email, message="some notification")
# # # # # # # # # # #     return {"message": "Notification sent in the background"}




# # # # # # # # # # from typing import Annotated

# # # # # # # # # # from fastapi import BackgroundTasks, Depends, FastAPI

# # # # # # # # # # app = FastAPI()


# # # # # # # # # # def write_log(message: str):
# # # # # # # # # #     with open("log.txt", mode="a") as log:
# # # # # # # # # #         log.write(message)


# # # # # # # # # # def get_query(background_tasks: BackgroundTasks, q: str | None = None):
# # # # # # # # # #     if q:
# # # # # # # # # #         message = f"found query: {q}\n"
# # # # # # # # # #         background_tasks.add_task(write_log, message)
# # # # # # # # # #     return q


# # # # # # # # # # @app.post("/send-notification/{email}")
# # # # # # # # # # async def send_notification(
# # # # # # # # # #     email: str, background_tasks: BackgroundTasks, q: Annotated[str, Depends(get_query)]
# # # # # # # # # # ):
# # # # # # # # # #     message = f"message to {email}\n"
# # # # # # # # # #     background_tasks.add_task(write_log, message)
# # # # # # # # # #     return {"message": "Message sent"}





# # # # # # # # # from typing import Annotated

# # # # # # # # # from fastapi import Depends, FastAPI
# # # # # # # # # from fastapi.security import OAuth2PasswordBearer

# # # # # # # # # app = FastAPI()

# # # # # # # # # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# # # # # # # # # @app.get("/items/")
# # # # # # # # # async def read_items(token: Annotated[str, Depends(oauth2_scheme)]):
# # # # # # # # #     return {"token": token}






# # # # # # # # from typing import Annotated

# # # # # # # # from fastapi import Depends, FastAPI, HTTPException, status
# # # # # # # # from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# # # # # # # # from pydantic import BaseModel

# # # # # # # # fake_users_db = {
# # # # # # # #     "johndoe": {
# # # # # # # #         "username": "johndoe",
# # # # # # # #         "full_name": "John Doe",
# # # # # # # #         "email": "johndoe@example.com",
# # # # # # # #         "hashed_password": "fakehashedsecret",
# # # # # # # #         "disabled": False,
# # # # # # # #     },
# # # # # # # #     "alice": {
# # # # # # # #         "username": "alice",
# # # # # # # #         "full_name": "Alice Wonderson",
# # # # # # # #         "email": "alice@example.com",
# # # # # # # #         "hashed_password": "fakehashedsecret2",
# # # # # # # #         "disabled": True,
# # # # # # # #     },
# # # # # # # # }

# # # # # # # # app = FastAPI()


# # # # # # # # def fake_hash_password(password: str):
# # # # # # # #     return "fakehashed" + password


# # # # # # # # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# # # # # # # # class User(BaseModel):
# # # # # # # #     username: str
# # # # # # # #     email: str | None = None
# # # # # # # #     full_name: str | None = None
# # # # # # # #     disabled: bool | None = None


# # # # # # # # class UserInDB(User):
# # # # # # # #     hashed_password: str


# # # # # # # # def get_user(db, username: str):
# # # # # # # #     if username in db:
# # # # # # # #         user_dict = db[username]
# # # # # # # #         return UserInDB(**user_dict)


# # # # # # # # def fake_decode_token(token):
# # # # # # # #     # This doesn't provide any security at all
# # # # # # # #     # Check the next version
# # # # # # # #     user = get_user(fake_users_db, token)
# # # # # # # #     return user


# # # # # # # # async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
# # # # # # # #     user = fake_decode_token(token)
# # # # # # # #     if not user:
# # # # # # # #         raise HTTPException(
# # # # # # # #             status_code=status.HTTP_401_UNAUTHORIZED,
# # # # # # # #             detail="Invalid authentication credentials",
# # # # # # # #             headers={"WWW-Authenticate": "Bearer"},
# # # # # # # #         )
# # # # # # # #     return user


# # # # # # # # async def get_current_active_user(
# # # # # # # #     current_user: Annotated[User, Depends(get_current_user)],
# # # # # # # # ):
# # # # # # # #     if current_user.disabled:
# # # # # # # #         raise HTTPException(status_code=400, detail="Inactive user")
# # # # # # # #     return current_user


# # # # # # # # @app.post("/token")
# # # # # # # # async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
# # # # # # # #     user_dict = fake_users_db.get(form_data.username)
# # # # # # # #     if not user_dict:
# # # # # # # #         raise HTTPException(status_code=400, detail="Incorrect username or password")
# # # # # # # #     user = UserInDB(**user_dict)
# # # # # # # #     hashed_password = fake_hash_password(form_data.password)
# # # # # # # #     if not hashed_password == user.hashed_password:
# # # # # # # #         raise HTTPException(status_code=400, detail="Incorrect username or password")

# # # # # # # #     return {"access_token": user.username, "token_type": "bearer"}


# # # # # # # # @app.get("/users/me")
# # # # # # # # async def read_users_me(
# # # # # # # #     current_user: Annotated[User, Depends(get_current_active_user)],
# # # # # # # # ):
# # # # # # # #     return current_user





# # # # # # # from datetime import datetime, timedelta, timezone
# # # # # # # from typing import Annotated

# # # # # # # import jwt
# # # # # # # from fastapi import Depends, FastAPI, HTTPException, status
# # # # # # # from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# # # # # # # from jwt.exceptions import InvalidTokenError
# # # # # # # from pwdlib import PasswordHash
# # # # # # # from pydantic import BaseModel

# # # # # # # # to get a string like this run:
# # # # # # # # openssl rand -hex 32
# # # # # # # SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
# # # # # # # ALGORITHM = "HS256"
# # # # # # # ACCESS_TOKEN_EXPIRE_MINUTES = 30


# # # # # # # fake_users_db = {
# # # # # # #     "johndoe": {
# # # # # # #         "username": "johndoe",
# # # # # # #         "full_name": "John Doe",
# # # # # # #         "email": "johndoe@example.com",
# # # # # # #         "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",
# # # # # # #         "disabled": False,
# # # # # # #     }
# # # # # # # }


# # # # # # # class Token(BaseModel):
# # # # # # #     access_token: str
# # # # # # #     token_type: str


# # # # # # # class TokenData(BaseModel):
# # # # # # #     username: str | None = None


# # # # # # # class User(BaseModel):
# # # # # # #     username: str
# # # # # # #     email: str | None = None
# # # # # # #     full_name: str | None = None
# # # # # # #     disabled: bool | None = None


# # # # # # # class UserInDB(User):
# # # # # # #     hashed_password: str


# # # # # # # password_hash = PasswordHash.recommended()

# # # # # # # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # # # # # # app = FastAPI()


# # # # # # # def verify_password(plain_password, hashed_password):
# # # # # # #     return password_hash.verify(plain_password, hashed_password)


# # # # # # # def get_password_hash(password):
# # # # # # #     return password_hash.hash(password)


# # # # # # # def get_user(db, username: str):
# # # # # # #     if username in db:
# # # # # # #         user_dict = db[username]
# # # # # # #         return UserInDB(**user_dict)


# # # # # # # def authenticate_user(fake_db, username: str, password: str):
# # # # # # #     user = get_user(fake_db, username)
# # # # # # #     if not user:
# # # # # # #         return False
# # # # # # #     if not verify_password(password, user.hashed_password):
# # # # # # #         return False
# # # # # # #     return user


# # # # # # # def create_access_token(data: dict, expires_delta: timedelta | None = None):
# # # # # # #     to_encode = data.copy()
# # # # # # #     if expires_delta:
# # # # # # #         expire = datetime.now(timezone.utc) + expires_delta
# # # # # # #     else:
# # # # # # #         expire = datetime.now(timezone.utc) + timedelta(minutes=15)
# # # # # # #     to_encode.update({"exp": expire})
# # # # # # #     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
# # # # # # #     return encoded_jwt


# # # # # # # async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
# # # # # # #     credentials_exception = HTTPException(
# # # # # # #         status_code=status.HTTP_401_UNAUTHORIZED,
# # # # # # #         detail="Could not validate credentials",
# # # # # # #         headers={"WWW-Authenticate": "Bearer"},
# # # # # # #     )
# # # # # # #     try:
# # # # # # #         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
# # # # # # #         username = payload.get("sub")
# # # # # # #         if username is None:
# # # # # # #             raise credentials_exception
# # # # # # #         token_data = TokenData(username=username)
# # # # # # #     except InvalidTokenError:
# # # # # # #         raise credentials_exception
# # # # # # #     user = get_user(fake_users_db, username=token_data.username)
# # # # # # #     if user is None:
# # # # # # #         raise credentials_exception
# # # # # # #     return user


# # # # # # # async def get_current_active_user(
# # # # # # #     current_user: Annotated[User, Depends(get_current_user)],
# # # # # # # ):
# # # # # # #     if current_user.disabled:
# # # # # # #         raise HTTPException(status_code=400, detail="Inactive user")
# # # # # # #     return current_user


# # # # # # # @app.post("/token")
# # # # # # # async def login_for_access_token(
# # # # # # #     form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
# # # # # # # ) -> Token:
# # # # # # #     user = authenticate_user(fake_users_db, form_data.username, form_data.password)
# # # # # # #     if not user:
# # # # # # #         raise HTTPException(
# # # # # # #             status_code=status.HTTP_401_UNAUTHORIZED,
# # # # # # #             detail="Incorrect username or password",
# # # # # # #             headers={"WWW-Authenticate": "Bearer"},
# # # # # # #         )
# # # # # # #     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
# # # # # # #     access_token = create_access_token(
# # # # # # #         data={"sub": user.username}, expires_delta=access_token_expires
# # # # # # #     )
# # # # # # #     return Token(access_token=access_token, token_type="bearer")


# # # # # # # @app.get("/users/me/", response_model=User)
# # # # # # # async def read_users_me(
# # # # # # #     current_user: Annotated[User, Depends(get_current_active_user)],
# # # # # # # ):
# # # # # # #     return current_user


# # # # # # # @app.get("/users/me/items/")
# # # # # # # async def read_own_items(
# # # # # # #     current_user: Annotated[User, Depends(get_current_active_user)],
# # # # # # # ):
# # # # # # #     return [{"item_id": "Foo", "owner": current_user.username}]







# # # # # # from fastapi import FastAPI
# # # # # # from fastapi.middleware.cors import CORSMiddleware

# # # # # # app = FastAPI()

# # # # # # origins = [
# # # # # #     "http://localhost.tiangolo.com",
# # # # # #     "https://localhost.tiangolo.com",
# # # # # #     "http://localhost",
# # # # # #     "http://localhost:8080",
# # # # # # ]

# # # # # # app.add_middleware(
# # # # # #     CORSMiddleware,
# # # # # #     allow_origins=origins,
# # # # # #     allow_credentials=True,
# # # # # #     allow_methods=["*"],
# # # # # #     allow_headers=["*"],
# # # # # # )


# # # # # # @app.get("/")
# # # # # # async def main():
# # # # # #     return {"message": "Hello World"}






# # # # # from fastapi import FastAPI, HTTPException

# # # # # app = FastAPI()

# # # # # items = {"foo": "The Foo Wrestlers"}


# # # # # @app.get("/items/{item_id}")
# # # # # async def read_item(item_id: str):
# # # # #     if item_id not in items:
# # # # #         raise HTTPException(status_code=404, detail="Item not found")
# # # # #     return {"item": items[item_id]}





# # # # from fastapi import FastAPI, Request
# # # # from fastapi.responses import JSONResponse


# # # # class UnicornException(Exception):
# # # #     def __init__(self, name: str):
# # # #         self.name = name


# # # # app = FastAPI()


# # # # @app.exception_handler(UnicornException)
# # # # async def unicorn_exception_handler(request: Request, exc: UnicornException):
# # # #     return JSONResponse(
# # # #         status_code=418,
# # # #         content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
# # # #     )


# # # # @app.get("/unicorns/{name}")
# # # # async def read_unicorn(name: str):
# # # #     if name == "yolo":
# # # #         raise UnicornException(name=name)
# # # #     return {"unicorn_name": name}\





# # # from fastapi import FastAPI, HTTPException
# # # from fastapi.exceptions import RequestValidationError
# # # from fastapi.responses import PlainTextResponse
# # # from starlette.exceptions import HTTPException as StarletteHTTPException

# # # app = FastAPI()


# # # @app.exception_handler(StarletteHTTPException)
# # # async def http_exception_handler(request, exc):
# # #     return PlainTextResponse(str(exc.detail), status_code=exc.status_code)


# # # @app.exception_handler(RequestValidationError)
# # # async def validation_exception_handler(request, exc):
# # #     return PlainTextResponse(str(exc), status_code=400)


# # # @app.get("/items/{item_id}")
# # # async def read_item(item_id: int):
# # #     if item_id == 3:
# # #         raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
# # #     return {"item_id": item_id}




# # from fastapi import FastAPI, Request
# # from fastapi.encoders import jsonable_encoder
# # from fastapi.exceptions import RequestValidationError
# # from fastapi.responses import JSONResponse
# # from pydantic import BaseModel

# # app = FastAPI()


# # @app.exception_handler(RequestValidationError)
# # async def validation_exception_handler(request: Request, exc: RequestValidationError):
# #     return JSONResponse(
# #         status_code=422,
# #         content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
# #     )


# # class Item(BaseModel):
# #     title: str
# #     size: int


# # @app.post("/items/")
# # async def create_item(item: Item):
# #     return item




# from fastapi import FastAPI, HTTPException
# from fastapi.exception_handlers import (
#     http_exception_handler,
#     request_validation_exception_handler,
# )
# from fastapi.exceptions import RequestValidationError
# from starlette.exceptions import HTTPException as StarletteHTTPException

# app = FastAPI()


# @app.exception_handler(StarletteHTTPException)
# async def custom_http_exception_handler(request, exc):
#     print(f"OMG! An HTTP error!: {repr(exc)}")
#     return await http_exception_handler(request, exc)


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     print(f"OMG! The client sent invalid data!: {exc}")
#     return await request_validation_exception_handler(request, exc)


# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     if item_id == 3:
#         raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
#     return {"item_id": item_id}