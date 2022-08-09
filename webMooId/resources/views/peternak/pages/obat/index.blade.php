@extends('peternak.layouts.app', ['activeMenu' => 'obat'])

@section('hstyles')

@endsection

@section('content')
<div class="pcoded-content">
    <!-- PAGE HEADER -->
    <div class="page-header">
        <div class="page-block">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="page-header-title">
                        <h5 class="m-b-10">Produk/Catalog Obat</h5>
                        <p class="m-b-0">Data Produk/Catalog Obat</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <ul class="breadcrumb">
                        <li class="breadcrumb-item">
                            <a href="#"><i class="fas fa-home"></i></a>
                        </li>
                        <li class="breadcrumb-item"><a href="#">Obat</a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    <!-- PAGE INNER CONTENT -->
    <div class="pcoded-inner-content">
        <div class="main-body">
            <div class="page-wrapper">
                <div class="page-body">
                    <!-- HERE IS THE EXAMPLE CONTENT-->
                    <div class="row">
                        <div class="col-3">
                            <a href="{{ route('obat.cart') }}">
                                <div class="card">
                                    <div class="card-block">
                                        <div class="row align-items-center m-l-0">
                                            <div class="col-auto">
                                                <i class="fas fa-cart-arrow-down f-30 text-c-green"></i>
                                            </div>
                                            <div class="col-auto">
                                                <h6 class="text-muted m-b-10">Keranjang Saya</h6>
                                                <h2 class="m-b-0">{{ (@$total_keranjang) ? $total_keranjang : '0' }}</h2>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </a>
                        </div>
                        <div class="col-9">
                            <div class="card card-main">
                                <div class="card-block">
                                    <form class="form-material" method="GET">
                                        <div class="row">
                                            <div class="col-6">
                                                <div class="form-group">
                                                    <div class="row">
                                                        <label class="col-sm-4 col-form-label">Berdasarkan:</label>
                                                        <div class="col-sm-8">
                                                            <select name="filter" class="form-control">
                                                                <option value="">Relevansi</option>
                                                                <option value="harga_termahal">Harga Termahal</option>
                                                                <option value="harga_termurah">Harga Termurah</option>
                                                            </select>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-6">
                                                <div class="right-icon-control">
                                                    <div class="form-group form-primary">
                                                        <input type="text" name="footer-email"
                                                            class="form-control">
                                                        <span class="form-bar"></span>
                                                        <label class="float-label">Search
                                                            here...</label>
                                                    </div>
                                                    <div class="form-icon">
                                                        <button
                                                            class="btn btn-success btn-icon  waves-effect waves-light">
                                                            <i class="fa fa-search"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-12">
                            <hr>
                        </div>
                    </div>
                    @if(@$info_result)
                    <div id="search_result" class="row">
                        <div class="col-12">
                            <h4 class="m-b-20"><b id="count_result">20</b> Search Results Found</h4>
                        </div>
                    </div>
                    @endif
                    <div class="row">
                        @if(@$info_obat)
                        @foreach($info_obat as $info_data)
                        @if($info_data->stok_obat->count() > 0)
                        <div class="col-xl-3 col-md-4 col-sm-6 col-12">
                            <div class="card prod-view">
                                <div class="prod-item text-center">
                                    <div class="prod-img">
                                        <div class="option-hover">
                                            <button type="button"
                                                class="btn btn-success btn-icon waves-effect waves-light m-r-15 hvr-bounce-in option-icon">
                                                <i class="icofont icofont-cart-alt f-20"></i>
                                            </button>
                                            <button type="button"
                                                class="btn btn-primary btn-icon waves-effect waves-light m-r-15 hvr-bounce-in option-icon">
                                                <i class="icofont icofont-eye-alt f-20"></i>
                                            </button>
                                            <button type="button"
                                                class="btn btn-danger btn-icon waves-effect waves-light hvr-bounce-in option-icon">
                                                <i class="icofont icofont-heart-alt f-20"></i>
                                            </button>
                                        </div>
                                        <a href="{{ route('obat.show', $info_data->id) }}" class="hvr-shrink">
                                            <img src="{{ (@$info_data->avatar) ? storage_path('public/obat/'. $info_data->id.'/'.$info_data->avatar) : asset('assets/images/product/p1.jpg') }}"
                                                class="img-fluid o-hidden">
                                        </a>
                                    </div>
                                    <div class="prod-info">
                                        <a href="{{ route('obat.show', $info_data->id) }}" class="txt-muted">
                                            <h4>{{ $info_data->nama_obat }}</h4>
                                        </a>
                                        <div class="m-b-10">
                                            <label class="label label-success">
                                                {{ $info_data->stok_obat->count() }} <i class="fab fa-gratipay"></i>
                                            </label>
                                            <a class="text-muted f-w-600">
                                                Tipe: {{ $info_data->tipe_obat }}
                                            </a>
                                        </div>
                                        <span class="prod-price">
                                            Rp. {{ $info_data->harga_obat }}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @endif
                        @endforeach
                        @endif
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
@endsection

@section('fscripts')

@endsection
