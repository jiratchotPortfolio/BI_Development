# Use an official Python runtime as a parent image
# เลือกใช้ Python 3.9 แบบ Slim (ขนาดเล็ก ประหยัดพื้นที่)
FROM python:3.9-slim

# Set work directory
# สร้างโฟลเดอร์ชื่อ app ไว้เก็บงานข้างใน
WORKDIR /app

# Install system dependencies
# ลงโปรแกรมพื้นฐานของ Linux ที่จำเป็นสำหรับการเชื่อมต่อ Database (PostgreSQL)
RUN apt-get update && apt-get install -y libpq-dev gcc

# Copy requirements and install dependencies
# ก๊อปปี้ไฟล์รายชื่อ Library และสั่งติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# ก๊อปปี้ไฟล์โปรเจกต์ทั้งหมดเข้าไป
COPY . .

# Define environment variable
# กำหนดค่า Port เริ่มต้น
ENV PORT=8080

# Run the ingestion script when the container launches
# สั่งให้รันไฟล์ ingestion.py ทันทีที่เปิดใช้งาน
CMD ["python", "src/ingestion.py"]
