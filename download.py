import os
import aiohttp
import asyncio

async def download_image(session, url, output_image_path):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(connect=0.5)) as response:
            if response.status == 200:
                with open(output_image_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(4096)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Image downloaded successfully as {output_image_path}")
            else:
                print(f"Failed to download {output_image_path}: HTTP status code {response.status}")
    except asyncio.TimeoutError:
        print(f"Skipping {output_image_path}: Connection timed out.")
    except aiohttp.ClientError as e:
        print(f"Skipping {output_image_path}: {e}")

async def download_images_in_parallel(root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vid_list = os.listdir(root)
    vid_image_paths = [os.path.join(root, vid, vid+'.url') for vid in vid_list]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for vid_image_path in vid_image_paths:
            with open(vid_image_path, 'r') as file:
                url = file.read().strip()
            file_prefix = os.path.splitext(os.path.basename(vid_image_path))[0]
            output_image_path = os.path.join(output_dir, f"{file_prefix}.png")
            tasks.append(download_image(session, url, output_image_path))
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    root = '/data1/JM/code/BrushNet/data/open_scene'
    output_dir = '/data1/JM/code/BrushNet/data/open_scene_image'
    asyncio.run(download_images_in_parallel(root, output_dir))
