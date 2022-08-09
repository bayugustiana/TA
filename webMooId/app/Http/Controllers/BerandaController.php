<?php

namespace App\Http\Controllers;

use App\Models\Beranda;

class BerandaController extends Controller
{

    public function welcome()
    {
        $data = array(
            'title' => 'Beranda'
        );
        return view('peternak.pages.welcome')->with($data);
    }

    public function index()
    {
        @$total_sapi = Beranda::total_sapi();
        @$total_tempat = Beranda::total_tempat();
        @$total_petugas = Beranda::total_petugas();
        @$status_ternak = Beranda::status_ternak();
        @$total_dokter = Beranda::total_dokter();
        @$total_artikel = Beranda::total_artikel();
        @$total_transaksi = Beranda::total_transaksi();
        @$total_toko_obat = Beranda::total_toko_obat();

        @$data_get = Beranda::tampil_mingguan();

        $data = array(
            'total_sapi' => $total_sapi,
            'total_tempat' => $total_tempat,
            'total_petugas' => $total_petugas,
            'status_ternak' => $status_ternak,
            'total_dokter' => $total_dokter,
            'total_artikel' => $total_artikel,
            'total_transaksi' => $total_transaksi,
            'total_toko_obat' => $total_toko_obat,
            'info_mingguan' => $data_get,
            'title' => 'Beranda'
        );
        return view('peternak.pages.beranda')->with($data);
    }
}
