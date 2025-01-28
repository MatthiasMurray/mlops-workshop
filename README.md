# mlops-workshop

Repository for educational workshop on building and deploying API endpoints for ML models to AWS

# Onboarding

1. To begin the workshop, fork this repo to your github profile (keep the name as mlops-workshop)
2. Let me know when you have forked the repo. I will run the `infra_setup/create_pipeline_service_roles.sh` and `infra_setup/create_pipelines.sh` on your username and github repo profile name
3. Once this is complete, go to the AWS portal and select the `mlops-workshop-student` role you will have been assigned access to
4. Go to the CodePipeline service, open the navigation bar on the left-hand side, click Settings>Connections
5. Select your connection. Your username was given to you when you signed up -- my username is `mmur`, so my connection is `mmur-connection`. Please do not touch other connections -- I've tried to limit access but things can slip through.
6. Click the `Update pending connection` button. A new browser window will pop up and should ask you to authorize GitHub access, or ask you to sign in first if you haven't in awhile in this browser. Once it has been approved, click the button to `Install new app` and approve GitHub once more -- **LIMIT ACCESS TO JUST YOUR FORKED REPO, DO NOT ALLOW ACCESS TO ALL** -- this creates a two-way connection, which is what we need. You can verify it was successful by going to your GitHub profile, clicking your profile icon (top right) and clicking the Settings button, then scrolling down to the bottom left and selecting Applications. You should see AWS Connector for GitHub under both the Installed GitHub Apps and Authorized GitHub Apps tabs.
7. Your CodePipeline is set up! Now when you merge to `main` on your forked repo, the build will trigger and create services with your username as a prefix

# Trying it out

1. Clone your forked repo down to your local machine
2. Open it in your preferred editor (VS Code is nice, but whatever you prefer)
3. Create a new branch -- if you wanted to name the branch `trying-it-out` you would open a new terminal and run `git checkout -b trying-it-out`
4. Make some change on this branch -- it could be as simple as a blank newline or funny comment added to this README, or you could modify the training procedure used for the text model trained in `services/textemote` by commenting out line 4 in `services/textemote/build.sh`. I added that line to reduce the time it took to run the pipeline, but it causes the model to finish training before gradient descent is really finished, so it hurts performance. Yours can be better!
5. Save your change, commit it (use the editor's git interface or run `git commit -a -m "My first commit for the workshop!"` in the terminal)
6. Push your change -- assuming you named your branch `trying-it-out` you'll run `git push origin trying-it-out`. You can ensure that your forked repo is the remote (rather than the original) by running `git remote -v` and making sure that `origin` is on your GitHub profile.
7. Go to your GitHub repo in the browser and click Pull Requests in the bar above the code. You want to create a pull request that merges the branch you pushed into your `main` branch.
8. Merge your branch into `main` if it looks good -- this should kick off your pipeline! You can watch its progress by opening up your CodePipeline in AWS and clicking View Details. You may need to refresh for it to continue to show what's going on. It takes about 5 minutes to complete in its current form, it could take up to 15 minutes if you comment out line 4 in `services/textemote/build.sh`.

# Testing your new service

1. If your pipeline finished successfully, the end of the logs should have printed out a URL where your service lives. Copy this URL, paste it into a browser, and add `/health` at the end. If it is running successfully, you should see `{'status': 'ok'}` in the browser. You can also go look at the deployment by opening up Elastic Container Service in AWS.
2. To test it, you can 'hit' it from your machine! This can be done a number of ways, it's publicly on the internet. Replace the url you pasted (with `/predict` on the end) with the dummy example provided:
   - From Terminal:
     ```
     curl -X POST "http://not-the-real-url/yourusername-textemote/predict" -H "Content-Type: application/json" -d "{'text': 'this is some text'}"
     ```
   - From Python:
     ```
     import requests
     url = "http://not-the-real-url/yourusername-textemote/predict"
     payload = {"text": "This is some text"}
     header = {"Content-Type": "application/json"}
     requests.post(url,header,payload)
     ```

# Next steps

Try adding your own service pattern! Make a copy of the `textemote` folder inside `services`, call it whatever you want (keep it short though).

If you want to just create a simple app to make sure you understand, try creating one that returns `{"answer": "helloworld"}` when you send a text payload.

The deployment pattern expects the scripts `setup.sh`, `build.sh`, and `deploy.sh` to exist, but you can just replace all the contents of `services/helloworld/setup.sh` with a simple print statement like `echo No setup needed, hello world!` and `services/helloworld/build.sh` with something similar like `echo No build needed, hello world!`

I'll leave it to you to figure out what you need to change in the `services/helloworld/Dockerfile`, `services/helloworld/app/main.py`, and `services/helloworld/requirements.txt`, but here's a hint -- there's no need for any of the spaCy stuff if you're just printing `{"answer": "helloworld"}` for any payload sent to `/predict`! Don't be too afraid to try crazy new things. Your `main` will not affect the rest of the code, and if your pipeline or services break it is unlikely to affect anyone elses ;D

# Local development

You can test these locally as well. To try out your new endpoint without deploying anything:

1. Create a virtual environment `python -m venv .venv`
2. Activate a virtual environment `source .venv/bin/activate` (or `.venv/Scripts/Activate.ps1` on Powershell, Windows can be tricky)
3. Install requirements `pip install -r requirements.txt`
4. Run the entrypoint command from the `Dockerfile`:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 80
   ```
   in your Terminal. This will open the app locally, and if you open a browser and go to `http://localhost:80/health` you should see your health check. Try using `curl` or Python locally to see if your `/predict` is working.
5. If you have Docker installed, try building the container. You will need to navigate to the `services/helloworld` folder, run:
   ```
   docker build --build-arg SERVICE_NAME=helloworld -t my-local-test-helloworld .
   ```
   to build an image called `my-local-test-helloworld`. You can run this with `docker run -p 80:80 my-local-test-helloworld`. This will start the Docker container, but probably won't show that it's left running the way directly running the app did -- you can check your Docker processes with `docker ps`, but it should otherwise work the same if you go to `http://localhost:80/health` or test it with `curl` or Python locally.

# Contributing

If you develop a new service (other than `helloworld`) that you think is cool, feel free to contribute to my repo! Once you've merged to your `main` and you're happy with the results:

1. Set up the contribution by configuring the remote to include my repo: `git remote add upstream <original-repo-url>`
2. You should now see the original repo if you run `git remote -v`.
3. If you only make changes in a new sub-folder of `service`, it's unlikely to conflict with the rest of the code. In that case, you can just run the following to make sure you have the most recent version of my repo's code included in your fork:
   `git fetch upstream`
   `git checkout main`
   `git merge upstream/main`
   `git push origin main`
4. Create a pull request from your fork to my repo using GitHub in the browser! I will review it and if I like it I'll approve and you'll be a contributor to an open source project!!

#Changing the stream to update the pipeline
