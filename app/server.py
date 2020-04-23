import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_url= 'https://drive.google.com/open?id=1-46Gpl8Llla1qWjHW817dRJiVFRu8Cub'
#fpath = '/Users/dexterdsilva/Documents/Developer/MachineLearning/fastai/course-v3/webapp/fastai-v3-master/app/models'
#export_file_name = fpath + '/22Apr_f1cars_export.pkl'
export_file_name = export_file_url + '22Apr_f1cars_export.pkl'
print(export_file_name)

classes = ['ferarri', 'mercedes', 'redbull', 'mclaren', 'racingpoint',
         'renault','williams', 'tororosso', 'haas', 'sauber']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    preds = learn.predict(img)[0]
    pred_idx = learn.predict(img)[1]
    probs = learn.predict(img)[2]
    confidence= str(round(probs.numpy()[pred_idx]*100)) + '%'
    #return JSONResponse({'result': str(prediction)})
    return JSONResponse({'result': str(preds),
                        'confidence': confidence})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
