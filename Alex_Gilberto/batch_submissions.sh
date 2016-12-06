for entry in "models"/*.yml
do
  echo config file "$entry"
  python generate_submission.py  -c "$entry" -p
done
