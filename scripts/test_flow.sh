JOB=$(curl -s -X POST http://localhost:8000/transcribe \
-F "audio=@data/samples/sample.wav" | jq -r .job_id)
echo "Job: $JOB"
# polling
while true; do
STATUS=$(curl -s http://localhost:8000/jobs/$JOB | jq -r .status)
echo "Status: $STATUS"
[ "$STATUS" = "done" ] && break
[ "$STATUS" = "failed" ] && exit 1
sleep 1
done
# fetch result
curl -s http://localhost:8000/jobs/$JOB/result | jq .full_text
