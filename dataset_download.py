from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="datasets")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test"])
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test"])