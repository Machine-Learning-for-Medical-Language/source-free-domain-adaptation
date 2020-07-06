competition/scoring_program.zip:scoring_program/*
	cd scoring_program && zip -r ../competition/scoring_program.zip * && cd ..

competition/practice_data.zip: practice_data/*
	cd practice_data && zip -r ../competition/practice_data.zip * && cd ..

competition/evaluation_data.zip: evaluation_data/*
	cd evaluation_data && zip -r ../competition/evaluation_data.zip * && cd ..

competition/post_evaluation_data.zip: post_evaluation_data/*
	cd post_evaluation_data && zip -r ../competition/post_evaluation_data.zip * && cd ..

competition.zip: competition/* competition/scoring_program.zip competition/practice_data.zip competition/evaluation_data.zip competition/post_evaluation_data.zip
	cd competition && zip ../competition.zip * && cd ..

submission.zip: submission/*
	cd submission && zip -r ../submission.zip * && cd ..