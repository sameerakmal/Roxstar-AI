import asyncio
import edge_tts

async def main():
    communicate = edge_tts.Communicate("Namaste! Main Roxstar AI Dost hoon.", "hi-IN-MadhurNeural")
    chunk_count = 0
    total_bytes = 0
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunk_count += 1
            total_bytes += len(chunk["data"])
    print(f"Edge-TTS success! Received {chunk_count} audio chunks, total {total_bytes} bytes.")

if __name__ == "__main__":
    asyncio.run(main())
