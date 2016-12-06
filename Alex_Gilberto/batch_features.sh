for entry in "config"/*.yml
do
  for subject in 1 2 3
  do
    echo config file "$entry" : subject "$subject"
    python generate_features.py -s "$subject" -n 8 -c "$entry" --old
  done
done
