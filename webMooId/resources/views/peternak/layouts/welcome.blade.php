<!-- Author : Aditiya Fadillah
Frontend Developer -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta content="initial-scale=1, minimum-scale=1, width=device-width" name="viewport">
    <link rel="shortcut icon" href="#" />
    <meta name="robots" content="all,follow">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <link rel="stylesheet" href="{{ asset('welcome/css/style.css') }}">
    <title>Moo.id | Home</title>
    <style>
        html {
            scroll-behavior: smooth;
        }
    </style>
</head>
<body>
    <!-- Full Page Image Header with Vertically Centered Content -->
    <header class="masthead" id="home">
        <div class="container h-100">
            <nav class="navbar navbar-expand-lg fixed-top" style="background-color: #3D9AE9;">
                <a class="navbar-brand" href="#">
                    <img src="{{ asset('welcome/assets/716569.png') }}" width="100" height="30" class="d-inline-block align-top" alt="">
                </a>
                <button type="button" class="navbar-toggler" data-toggle="collapse" data-target="#navbarCollapse">
                    <span class="navbar-toggler-icon"><img src="{{ asset('welcome/assets/open-menu.png')  }}" width="30" height="30" class="d-inline-block align-top" alt=""></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarCollapse">
                    <ul class="navbar-nav ml-auto">
                        <li class="nav-item active">
                            <a class="nav-link text-white" href="#home">Home <span class="sr-only">(current)</span></a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="#about">About</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="#fitur">Fitur</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="#team">Team</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link text-white" href="#contact">Contact Us</a>
                        </li>
                    </ul>
                </div>
            </nav>
            <div class="row h-100 align-items-center" id="tagline">
                <div class="col-md-6 col-sm-12">
                    <div class="big-tagline text-white" id="tagline-text">Easy Solution</div>
                    <div class="display-4 text-white" id="tagline-text-2">for Animal Farming</div><br><br>
                    <div id="img-playstore">
                        <a href="#"><img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png" width="140" height="60"></a>
                    </div>
                </div>
                <div class="col-md-6 col-sm-12" id="tagline-img">
                    <img src="{{ asset('welcome/assets/apps.png') }}" class="img-fluid" alt="">
                </div>
            </div>
        </div>
    </header>

    @yield('content')

    <!-- footer -->
    <footer class="container-fluid bg-grey py-5" id="contact">
        <div class="container">
           <div class="row">
              <div class="col-md-12">
                 <div class="row">
                    <div class="col-md-6 ">
                       <div class="logo-part">
                          <img src="{{ asset('welcome/assets/716569.png') }}" class="w-50 logo-footer" >
                          <p><img src="{{ asset('welcome/assets/lokasi.png') }}" width="30px">&nbsp;Jl. Dipati Ukur No.112-116, 
                            Lebakgede, Kecamatan Coblong, Kota Bandung, Jawa Barat 40132</p>
                          <p><img src="{{ asset('welcome/assets/phone.png') }}" width="30px">&nbsp;+62 85759066401</p>
                          <a href="oficialymooid@gmail.com"><p><img src="{{ asset('welcome/assets/email.png') }}" width="30px">&nbsp;oficialymooid@gmail.com</p></a>
                       </div>
                    </div>
                    <div class="col-md-6 px-4">
                       <h6> About Company</h6>
                       <p class="text">Banyak sekali permasalahan di dunia peternakan khususnya di Indonesia dan tingginya permintaan pasar untuk hasil produksi di bidang peternakan . Moo.id lahir untuk membantu lebih dari +4.000.000 peternak di seluruh Indonesia untuk meningkatkan hasil produksi demi kebutuhan pasar. Dengan teknologi terbaru, kami dapat memantau kondisi kesehatan ternak untuk mempermudah kerja para peternak di seluruh Indonesia .</p>
                       <a href="#" class="btn-footer"> More Info </a><br>
                       <a href="#" class="btn-footer"> Contact Us</a>
                    </div>
                 </div>
              </div>
           </div>
        </div>
    </footer>
    <!-- penutup footer -->
</div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</body>
</html>