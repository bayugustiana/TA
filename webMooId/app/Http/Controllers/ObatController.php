<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Obat;
use App\Models\Toko_Obat;
use App\Models\Transaksi;

class ObatController extends Controller
{
    public function index()
    {
        $data_get = Obat::all();
        $data = array(
            'info_obat' => $data_get,
            'title' => 'Catalog Obat'
        );
        return view('peternak.pages.obat.index')->with($data);
    }

    public function show($id)
    {
        $data_get = Obat::find($id);
        $data = array(
            'info' => $data_get,
            'title' => 'Lihat Obat #'.$id,
        );
        return view('peternak.pages.obat.show')->with($data);
    }

    public function cart()
    {
        $data_get = Transaksi::where('id_peternakan','1')->where('status','keranjang')->get();
        $data = array(
            'info_transaksi' => $data_get,
            'title' => 'Keranjang Obat Saya',
        );
        return view('peternak.pages.obat.cart')->with($data);
    }

    public function track()
    {
        $data_get = Transaksi::where('id_peternakan','1')->where('status', 'dikonfirmasi')->get();
        $data_get2 = Transaksi::where('id_peternakan','1')->where('status', 'diproses')->get();
        $data = array(
            'info_transaksi_dikonfirmasi' => $data_get,
            'info_transaksi_diproses' => $data_get2,
            'title' => 'Lacak Pesanan Obat Saya',
        );
        return view('peternak.pages.obat.track')->with($data);
    }
    public function show_toko_obat($id)
    {
        $data_get = Toko_Obat::find($id);
        $data_get2 = $data_get->obat;
        $data = array(
            'info' => $data_get,
            'info_obat' => $data_get2,
            'title' => 'Lihat Toko Obat #'.$data_get->nama_toko_obat,
        );
        return view('peternak.pages.obat.toko')->with($data);
    }
}
