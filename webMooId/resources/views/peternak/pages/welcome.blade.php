@extends('peternak.layouts.welcome')

@section('content')
    <!-- Page Content -->
    <section class="jumbotron jumbotron-fluid" id="about">
        <div class="container">
            <div class="row">
                <div class="col-md-8 col-sm-12">
                    <h2 class="display-5 text-black" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">About</h2>
                    <hr>
                    <p class="text">Banyak sekali permasalahan di dunia peternakan khususnya di Indonesia dan tingginya permintaan pasar untuk hasil produksi di bidang peternakan . Moo.id lahir untuk membantu lebih dari +4.000.000 peternak di seluruh Indonesia untuk meningkatkan hasil produksi demi kebutuhan pasar. Dengan teknologi terbaru, kami dapat memantau kondisi kesehatan ternak untuk mempermudah kerja para peternak di seluruh Indonesia .
                    </p>
                </div>
                <div class="col-md-4 col-sm-12 text-center-fluid">
                    <img src="{{ asset('welcome/assets/logo.png') }}" class="img-responsive" alt="">
                </div>
            </div>
        </div>
    </section>

    <section id="fitur">
        <div class="container">
            <div class="row">
                <div class="col-md-8 col-sm-12">
                    <h2 class="display-5 text-black" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: bold;">Fitur</h2>
                    <hr>
                </div>
            </div>
            <div class="row justify-content-start">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <div class="media-body">
                            <h3 class="mt-0">Pelacak Ternak</h3>
                            <p>Selalu mengetahui posisi ternak secara aktual</p>
                        </div>
                        <img class="align-self-center ml-3 iconic-image" src="{{ asset('welcome/assets/kacamata.png') }}" alt="Generic placeholder image">
                    </div>
                </div>
            </div>
            <div class="row justify-content-end">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <img class="align-self-center mr-3 iconic-image" src="{{ asset('welcome/assets/calender.png') }}" alt="Generic placeholder image">
                        <div class="media-body">
                            <h3 class="mt-0">Terintegrasi 24Jam</h3>
                            <p>dapat terhubung dengan system selama 24Jam</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row justify-content-start">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <div class="media-body">
                            <h3 class="mt-0">Machine Learning</h3>
                            <p>Dapat melakukan analisis dan prediksi dari data yang tersedia di cloud.</p>
                        </div>
                        <img class="align-self-center ml-3 iconic-image" src="{{ asset('welcome/assets/microchip.png') }}" alt="Generic placeholder image">
                    </div>
                </div>
            </div>
            <div class="row justify-content-end">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <img class="align-self-center mr-3 iconic-image" src="{{ asset('welcome/assets/medical-15_icon-icons.com_73938.png') }}" alt="Generic placeholder image">
                        <div class="media-body">
                            <h3 class="mt-0">Rekam Medis</h3>
                            <p>Riwayat penyakit dan kecelakaan hewan ternak akan terekap</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row justify-content-start">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <div class="media-body">
                            <h3 class="mt-0">Notifikasi Kondisi Ternak</h3>
                            <p>Sistem akan mengecek kondisi ternak dan menganalisis untuk memberitahukan kepada pemilik ternak.</p>
                        </div>
                        <img class="align-self-center ml-3 iconic-image" src="{{ asset('welcome/assets/notifikasi.png') }}" alt="Generic placeholder image">
                    </div>
                </div>
            </div>
            <div class="row justify-content-end">
                <div class="col-md-6 col-sm-12">
                    <div class="media">
                        <img class="align-self-center mr-3 iconic-image" src="{{ asset('welcome/assets/doctor.png') }}" alt="Generic placeholder image">
                        <div class="media-body">
                            <h3 class="mt-0">Konsultasi Dokter</h3>
                            <p>Peternak dapat berkonsultasi kepada hewan ternak mulai dari gejala penyakit sampai penyembuhan</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    
        <!-- Our Team -->
    <section id="team" class="pb-5">
      <div class="container">
          <h1 class="section-title h1">Work with us</h1>
          <div class="row">
              <!-- Team member -->
              <div class="col-xs-12 col-sm-6 col-md-4">
                  <!-- <div class="image-flip" ontouchstart="this.classList.toggle('hover');">
                      <div class="mainflip">
                          <div class="frontside">
                              <div class="card">
                                  <div class="card-body text-center">
                                      <p><img class=" img-fluid" src="{{ asset('welcome/assets/lurde.png') }}" alt="card image"></p>
                                      <h4 class="card-title">Moh Ripan SM</h4>
                                      <p class="card-text">Data Scientist</p>
                                  </div>
                              </div>
                          </div>
                          <div class="backside">
                              <div class="card">
                                  <div class="card-body text-center mt-4">
                                      <h4 class="card-title">Moh Ripan SM</h4>
                                      <p class="card-text">Disini saya sebagai Data Scientist untuk analisis data.</p>
                                      <ul class="list-inline">
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-facebook"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-twitter"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-skype"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-google"></i>
                                              </a>
                                          </li>
                                      </ul>
                                  </div>
                              </div>
                          </div>
                      </div>
                  </div> -->
              </div>
              <!-- ./Team member -->
              <!-- Team member -->
              <div class="col-xs-12 col-sm-6 col-md-4">
                  <div class="image-flip" ontouchstart="this.classList.toggle('hover');">
                      <div class="mainflip">
                          <div class="frontside">
                              <div class="card">
                                  <div class="card-body text-center">
                                      <p><img class=" img-fluid" src="{{ asset('welcome/assets/bayu.jpeg')  }}"></p>
                                      <h4 class="card-title">Bayu Gustiana Fajar</h4>
                                      <p class="card-text">IoT</p>
                                  </div>
                              </div>
                          </div>
                          <div class="backside">
                              <div class="card">
                                  <div class="card-body text-center mt-4">
                                      <h4 class="card-title">Bayu Agustina F</h4>
                                      <p class="card-text">Disini saya sebagai IoT development yang bertugas merancang dibagian electrical</p>
                                      <ul class="list-inline">
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-facebook"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-twitter"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-skype"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-google"></i>
                                              </a>
                                          </li>
                                      </ul>
                                  </div>
                              </div>
                          </div>
                      </div>
                  </div>
              </div>
              <!-- ./Team member -->
              <!-- Team member -->
              <div class="col-xs-12 col-sm-6 col-md-4">
                  <!-- <div class="image-flip" ontouchstart="this.classList.toggle('hover');">
                      <div class="mainflip">
                          <div class="frontside">
                              <div class="card">
                                  <div class="card-body text-center">
                                      <p><img class=" img-fluid" src="{{ asset('welcome/assets/rafi.png') }}" alt="card image"></p>
                                      <h4 class="card-title">M Rafi</h4>
                                      <p class="card-text">Web Developer</p>
                                  </div>
                              </div>
                          </div>
                          <div class="backside">
                              <div class="card">
                                  <div class="card-body text-center mt-4">
                                      <h4 class="card-title">M Rafi</h4>
                                      <p class="card-text">Disini saya sebagai web developer dari Moo.id, tugas saya adalah menerima data dari alat yang kemudian diolah dan dianalisis oleh Ripan dibagian Data Scientist</p>
                                      <ul class="list-inline">
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-facebook"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-twitter"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-skype"></i>
                                              </a>
                                          </li>
                                          <li class="list-inline-item">
                                              <a class="social-icon text-xs-center" target="_blank" href="#">
                                                  <i class="fa fa-google"></i>
                                              </a>
                                          </li>
                                      </ul>
                                  </div>
                              </div>
                          </div>
                      </div>
                  </div> -->
              </div>
              <!-- ./Team member -->
    
          </div>
      </div>
    </section>
    <!-- penutup team -->
@endsection